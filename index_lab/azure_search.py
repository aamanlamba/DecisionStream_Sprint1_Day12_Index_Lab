"""
azure_search.py — the real index. Azure AI Search over REST, stdlib only.

This is the file the whole lab has been pointing at. Everything the simulated
index does by hand, a real service does here, and the mapping is one to one:

    lab                             Azure AI Search
    ------------------------------  --------------------------------------
    VectorIndex._passes()           an OData $filter on filterable fields
    mode="keyword"                  "search": <text>          (BM25)
    mode="vector"                   "vectorQueries": [...]    (HNSW / cosine)
    mode="hybrid"                   both, fused by RRF
    filter-before-similarity        "vectorFilterMode": "preFilter"

THE ONE THAT IS WORTH THE WHOLE SESSION
---------------------------------------
`vectorFilterMode` has THREE settings, and they are not "the same answer
arriving by different routes" — they are different answers. Per the Microsoft
Learn page "Vector query filters" (checked 2026-08-12):

    preFilter          apply the filter DURING HNSW traversal on each shard,
                       expanding the graph until k candidates are found.
                       Guarantees k results if they exist. The default for
                       indexes created after ~15 Oct 2023, and recommended.

    postFilter         traverse each shard WITHOUT the filter to get that
                       shard's local top-k, then filter that local top-k,
                       then aggregate. Misses matches per shard.

    strictPostFilter   find the UNFILTERED GLOBAL top-k first, then filter it.
                       (preview — needs a preview api-version.) Highest risk
                       of false negatives; can return ZERO results even when
                       matches exist.

MEASURED ON THIS CORPUS — read this before you teach it
--------------------------------------------------------
    filter jurisdiction eq 'IE' (5 of 45 chunks), k=3

    preFilter          3 results
    postFilter         3 results     <- INDISTINGUISHABLE at this size
    strictPostFilter   0 results     <- the failure, in full

`postFilter` is a no-op here and that is not a bug. It degrades PER SHARD —
it filters each shard's local top-k — so producing the divergence needs two
things this lab does not have:

  * enough documents that a shard's local top-k is a real subset. 45 is not.
    Microsoft's own benchmark calls the two modes "approximately equal" on
    small indexes and puts the divergence at 10^5-10^6 documents.

  * enough SHARDS. Shard count follows PARTITION count, and that is a tier
    property: Basic ships with 1 partition (3 maximum), against 12 on
    S1/S2/S3. On a single-partition Basic service the multi-shard behaviour
    postFilter degrades on is largely out of reach at ANY corpus size.

So on Basic this is not a "small corpus" caveat you can grow out of. Teach
the mode; do not claim to have measured it.

strictPostFilter, by contrast, filters the GLOBAL top-k, so it fails on a
corpus of any size — which is why it is the one that demonstrates the point
here. Do not generalise its 0/10 into a claim about postFilter: they are
different modes with different failure profiles.

FIELDS ARE A DESIGN-TIME DECISION
---------------------------------
`filterable` is set when the index is CREATED. You cannot filter on a field
you did not mark, and marking it later means dropping the index, rebuilding
it and re-embedding every chunk. That re-embed is the one step in this
pipeline that costs real money at scale, which is why the metadata schema is
an architecture decision and not a detail.

DOCUMENT KEYS
-------------
The lab's chunk ids look like `POL-MOT-UK-2026#s-0`. Azure document keys allow
only letters, digits, underscore, dash and equals — `#` and `.` are rejected,
and the error you get names the document, not the rule. So keys are sanitised
here and the original is kept in a retrievable `chunk_id` field. Every real
ingestion pipeline grows this function.

VERIFY THE API VERSION. Verified against 2026-04-01 (current stable) on
2026-08-12. `vectorQueries` was itself a breaking change from the 2023-07-01
preview shape, so assume it will move again.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import env as _env
from indexing import Chunk

VECTOR_FIELD = "contentVector"
PROFILE = "policy-hnsw-profile"
ALGORITHM = "policy-hnsw"

_KEY_OK = re.compile(r"[^A-Za-z0-9_\-=]")


class SearchError(RuntimeError):
    pass


def sanitise_key(chunk_id: str) -> str:
    """Azure document keys: letters, digits, _, - and = only."""
    return _KEY_OK.sub("_", chunk_id)


def _iso(date_str):
    """'2026-01-01' -> '2026-01-01T00:00:00Z'. Edm.DateTimeOffset needs the
    offset; a bare date is rejected."""
    if not date_str:
        return None
    return "{}T00:00:00Z".format(date_str)


# ==========================================================================
# THE FILTER — the part to get right at design time
# ==========================================================================
def odata_filter(flt) -> str:
    """
    Translate the lab's filter dict into an OData expression.

    The date clause is the interesting one. `effective_to` is null for the
    version in force, and in OData a null NEVER satisfies a comparison — so
    `effective_to ge <date>` silently excludes the current policy, which is
    the answer you wanted. It must be spelled:

        (effective_to eq null or effective_to ge <date>)

    That single missing null check is a very common production bug in exactly
    this kind of index, and it fails by returning a confident wrong answer
    rather than an error.
    """
    if not flt:
        return None
    parts = []
    if flt.get("jurisdiction"):
        parts.append("jurisdiction eq '{}'".format(_esc(flt["jurisdiction"])))
    if flt.get("policy_type"):
        parts.append("policy_type eq '{}'".format(_esc(flt["policy_type"])))
    on = flt.get("in_force_on")
    if on:
        d = _iso(on)
        parts.append("effective_from le {}".format(d))
        parts.append("(effective_to eq null or effective_to ge {})".format(d))
    return " and ".join(parts) if parts else None


def _esc(s: str) -> str:
    return s.replace("'", "''")


# ==========================================================================
# THE INDEX SCHEMA
# ==========================================================================
def index_schema(name: str, dims: int) -> dict:
    return {
        "name": name,
        "fields": [
            # Key. Sanitised chunk id.
            {"name": "id", "type": "Edm.String", "key": True,
             "filterable": True, "retrievable": True, "searchable": False},

            # The original id, kept so the lab can report the same chunk names
            # the offline run does.
            {"name": "chunk_id", "type": "Edm.String", "retrievable": True,
             "searchable": False, "filterable": False},

            {"name": "doc_id", "type": "Edm.String", "retrievable": True,
             "filterable": True, "facetable": True, "searchable": False},

            {"name": "title", "type": "Edm.String", "retrievable": True,
             "searchable": True, "filterable": False},

            # The human-readable chunk. BM25 runs over this, and it is what
            # the judge inspects for the expected wording.
            {"name": "content", "type": "Edm.String", "retrievable": True,
             "searchable": True, "analyzer": "en.microsoft"},

            # ---- THE FILTERABLE METADATA. This is the lesson. ----
            {"name": "policy_type", "type": "Edm.String", "retrievable": True,
             "filterable": True, "facetable": True, "searchable": False},
            {"name": "jurisdiction", "type": "Edm.String", "retrievable": True,
             "filterable": True, "facetable": True, "searchable": False},
            {"name": "effective_from", "type": "Edm.DateTimeOffset",
             "retrievable": True, "filterable": True, "sortable": True},
            {"name": "effective_to", "type": "Edm.DateTimeOffset",
             "retrievable": True, "filterable": True, "sortable": True},
            {"name": "superseded", "type": "Edm.Boolean", "retrievable": True,
             "filterable": True, "facetable": True},
            {"name": "version", "type": "Edm.String", "retrievable": True,
             "filterable": True, "searchable": False},

            # ---- THE VECTOR ----
            # filterable/sortable/facetable MUST be false on a vector field.
            # stored=false drops the extra retrievable copy: the vector is
            # still searchable, it just is not returned. On this corpus it is
            # a rounding error; at a million chunks it is most of your bill.
            {"name": VECTOR_FIELD, "type": "Collection(Edm.Single)",
             "searchable": True, "retrievable": False, "stored": False,
             "dimensions": dims, "vectorSearchProfile": PROFILE},
        ],
        "vectorSearch": {
            "algorithms": [{
                "name": ALGORITHM,
                "kind": "hnsw",
                # cosine, because that is what Azure OpenAI embeddings use.
                # Choosing a metric the model was not trained for produces a
                # ranking that looks plausible and is wrong.
                "hnswParameters": {"m": 4, "efConstruction": 400,
                                   "efSearch": 500, "metric": "cosine"},
            }],
            "profiles": [{"name": PROFILE, "algorithm": ALGORITHM}],
        },
    }


# ==========================================================================
# THE CLIENT
# ==========================================================================
class AzureSearchIndex:

    def __init__(self, endpoint, api_key, index_name, api_version,
                 embedder=None, dims=None, preview_api_version=None):
        self.endpoint = _env.normalise_search_endpoint(endpoint)
        self.api_key = api_key
        self.name = index_name
        self.api_version = api_version
        # strictPostFilter is preview-only: the stable version rejects it with
        # a 400. Rather than force every call onto a preview contract, the
        # client uses the preview version ONLY for the queries that need it.
        self.preview_api_version = preview_api_version or "2026-05-01-preview"
        self.embedder = embedder
        self.dims = dims or (embedder.dims if embedder else 1536)
        self.query_calls = 0
        # Populated by provision()/upload() so the report can quote them.
        self.chunks = []
        self.embed_tokens = 0

    # ---------------- HTTP ----------------
    def _url(self, path: str, api_version=None, **params) -> str:
        params["api-version"] = api_version or self.api_version
        return "{}{}?{}".format(self.endpoint, path, urllib.parse.urlencode(params))

    def _call(self, method, path, body=None, timeout=90, api_version=None, **params):
        url = self._url(path, api_version=api_version, **params)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("api-key", self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode()
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:1200]
            raise SearchError("HTTP {} {} {}\n{}".format(e.code, method, url, detail))
        except urllib.error.URLError as e:
            raise SearchError("Could not reach {}\n  {}".format(url, e.reason))

    # ---------------- provisioning ----------------
    def exists(self) -> bool:
        try:
            self._call("GET", "/indexes/" + self.name)
            return True
        except SearchError as e:
            if "HTTP 404" in str(e):
                return False
            raise

    def drop(self):
        try:
            self._call("DELETE", "/indexes/" + self.name)
            return True
        except SearchError as e:
            if "HTTP 404" in str(e):
                return False
            raise

    def create(self, recreate: bool = True):
        """PUT is create-or-update. A dimensions change is NOT an update the
        service will accept in place — it needs the index dropped. That is
        the re-index cost, made concrete."""
        if recreate and self.exists():
            self.drop()
            time.sleep(2)
        return self._call("PUT", "/indexes/" + self.name,
                          body=index_schema(self.name, self.dims),
                          allowIndexDowntime="true")

    # ---------------- loading ----------------
    def upload(self, chunks, progress=None, batch: int = 100):
        """Embed the chunks and push them. Returns the number uploaded."""
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        vectors = self.embedder.embed_all(texts, dims=self.dims,
                                          progress=progress)
        self.embed_tokens = self.embedder.tokens_used

        docs = []
        for c, v in zip(self.chunks, vectors):
            m = c.meta
            docs.append({
                "@search.action": "upload",
                "id": sanitise_key(c.id),
                "chunk_id": c.id,
                "doc_id": c.doc_id,
                "title": m.get("title", ""),
                "content": c.text,
                "policy_type": m["policy_type"],
                "jurisdiction": m["jurisdiction"],
                "effective_from": _iso(m["effective_from"]),
                "effective_to": _iso(m["effective_to"]),
                "superseded": bool(m["superseded"]),
                "version": m.get("version", ""),
                VECTOR_FIELD: v,
            })

        uploaded = 0
        for i in range(0, len(docs), batch):
            resp = self._call("POST", "/indexes/{}/docs/index".format(self.name),
                              body={"value": docs[i:i + batch]})
            for r in resp.get("value", []):
                if not r.get("status"):
                    raise SearchError("Upload rejected {}: {}".format(
                        r.get("key"), r.get("errorMessage")))
                uploaded += 1
        return uploaded

    def service_stats(self) -> dict:
        """GET /servicestats — usage against your REAL quotas.

        This is a data-plane call, so it works with the same admin key and
        without any ARM/management-plane permission. On a locked-down learner
        subscription where `az` cannot list resources, this is how you find
        out what tier you are actually on: the quota numbers identify it.

            indexes 15 + storage 15 GB + vector 5 GB  ->  Basic (post-Apr-2024)
            indexes 50 + storage 160 GB + vector 35 GB ->  S1

        The vector quota is the one that bites first on a real corpus, and it
        is the reason `--dims` is a cost lever and not a curiosity.
        """
        return self._call("GET", "/servicestats")

    def capacity_note(self) -> str:
        """One line of honest capacity arithmetic from live numbers."""
        try:
            c = self.service_stats().get("counters", {})
        except SearchError:
            return ""
        vec = c.get("vectorIndexSize", {})
        used, quota = vec.get("usage") or 0, vec.get("quota") or 0
        idxs = c.get("indexesCount", {})
        n = c.get("documentCount", {}).get("usage") or 0
        if not (used and quota and n):
            return ""
        per = used / n
        return ("  quota       : {:.2f} MB of {:,.0f} MB vector ({:.3f}% used), "
                "indexes {}/{}\n"
                "                {:,.0f} B per chunk at {} dims -> this tier "
                "holds ~{:,} chunks".format(
                    used / 1e6, quota / 1e6, used / quota * 100,
                    idxs.get("usage"), idxs.get("quota"),
                    per, self.dims, int(quota / per)))

    def doc_count(self) -> int:
        url = self._url("/indexes/{}/docs/$count".format(self.name))
        req = urllib.request.Request(url, method="GET")
        req.add_header("api-key", self.api_key)
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.read().decode().strip().lstrip("﻿"))

    def wait_until_indexed(self, expected: int, timeout: int = 90) -> int:
        """Indexing is asynchronous. Querying immediately after upload is the
        classic flaky-test cause: the documents are accepted and not yet
        searchable."""
        deadline = time.time() + timeout
        n = 0
        while time.time() < deadline:
            n = self.doc_count()
            if n >= expected:
                return n
            time.sleep(2)
        return n

    # ---------------- querying ----------------
    def search(self, query, top_k, flt=None, mode="hybrid",
               vector_filter_mode="preFilter", exhaustive=True):
        """Returns [(Chunk, score)] so the lab's judge works unchanged."""
        body = {
            "top": top_k,
            "select": "id,chunk_id,doc_id,title,content,policy_type,"
                      "jurisdiction,effective_from,effective_to,superseded,version",
        }
        f = odata_filter(flt)
        if f:
            body["filter"] = f

        if mode in ("keyword", "hybrid"):
            body["search"] = query
            body["queryType"] = "simple"
            body["searchFields"] = "content,title"

        if mode in ("vector", "hybrid"):
            qv = self.embedder.embed(query, dims=self.dims)
            body["vectorQueries"] = [{
                "kind": "vector",
                "vector": qv,
                "fields": VECTOR_FIELD,
                "k": top_k,
                # On ~45 chunks HNSW's approximation is noise. exhaustive
                # makes the lab reproducible. In production you want the
                # approximation — that is what you are paying HNSW for.
                "exhaustive": exhaustive,
            }]
            body["vectorFilterMode"] = vector_filter_mode

        # strictPostFilter exists only on the preview contract.
        apiv = (self.preview_api_version
                if vector_filter_mode == "strictPostFilter" else None)
        resp = self._call("POST", "/indexes/{}/docs/search".format(self.name),
                          body=body, api_version=apiv)
        self.query_calls += 1

        out = []
        for hit in resp.get("value", []):
            out.append((_to_chunk(hit), float(hit.get("@search.score", 0.0))))
        return out

    def candidate_count(self, flt=None) -> int:
        """How many chunks survive the filter. The number that explains the
        whole result table."""
        body = {"search": "*", "top": 0, "count": True}
        f = odata_filter(flt)
        if f:
            body["filter"] = f
        resp = self._call("POST", "/indexes/{}/docs/search".format(self.name),
                          body=body)
        return int(resp.get("@odata.count", 0))


def _to_chunk(hit) -> Chunk:
    """Rebuild the lab's Chunk from a search result so nothing downstream —
    the judge, the report, the per-query CSV — needs to know this row came
    from Azure rather than from the offline index."""
    return Chunk(
        id=hit.get("chunk_id") or hit.get("id"),
        doc_id=hit.get("doc_id", ""),
        text=hit.get("content", ""),
        n_tokens=max(1, len(hit.get("content", "")) // 4),
        meta={
            "id": hit.get("doc_id", ""),
            "title": hit.get("title", ""),
            "policy_type": hit.get("policy_type", ""),
            "jurisdiction": hit.get("jurisdiction", ""),
            "effective_from": (hit.get("effective_from") or "")[:10],
            "effective_to": (hit.get("effective_to") or "")[:10] or None,
            "version": hit.get("version", ""),
            "superseded": hit.get("superseded", False),
        },
    )


def resolve_search() -> dict:
    return {
        "endpoint": _env.normalise_search_endpoint(
            _env.get("AZURE_SEARCH_ENDPOINT", required=True,
                     hint="Portal > your Search service > Overview > Url.")),
        "api_key": _env.get("AZURE_SEARCH_API_KEY", required=True,
                            hint="Portal > Search service > Settings > Keys > "
                                 "Primary ADMIN key (a query key cannot create "
                                 "an index)."),
        "index": _env.get("AZURE_SEARCH_INDEX", "decisionstream-policy"),
        "api_version": _env.get("AZURE_SEARCH_API_VERSION", "2026-04-01"),
        # Used ONLY for strictPostFilter queries, which the stable contract
        # rejects with a 400.
        "preview_api_version": _env.get("AZURE_SEARCH_PREVIEW_API_VERSION",
                                        "2026-05-01-preview"),
    }
