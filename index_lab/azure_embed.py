"""
azure_embed.py — real Azure OpenAI embeddings, drop-in for the simulated ones.

WHAT THIS SWAPS
---------------
`indexing.EmbeddingModel` hashes words into a vector. This class calls
text-embedding-3-* instead. Everything else in the lab — chunking, the
filterable index, the judge, the report — is UNCHANGED. That is the point:
you change exactly one thing and read what it did to each lever.

The prediction the lab makes is falsifiable, so go and falsify it:

    the MODEL lever should move (real vectors handle paraphrase)
    the FILTER lever should NOT move (no embedding model knows your
                                      jurisdiction)

THREE THINGS WORTH READING THE CODE FOR
---------------------------------------
1. TWO ROUTES. Azure serves embeddings at two different URLs and not every
   resource serves both. Guessing wrong 404s on every deployment name you
   own. `probe()` tries both and tells you which one this resource speaks.

2. THE CACHE IS THE COST CONTROL. The grid runs 54 configurations. Embedding
   the corpus once per (chunker, model) and caching to disk is the difference
   between one API pass and fifty-four. Real indexing pipelines cache for the
   same reason, and it is why a metadata schema change that forces a re-embed
   is the expensive kind of mistake.

3. TRUNCATION IS CLIENT-SIDE, ON PURPOSE. text-embedding-3 is Matryoshka-
   trained: the first N components of the full vector are themselves a usable
   embedding. So `--dims` truncates the CACHED full vector and renormalises,
   rather than paying for eight more API passes. Azure will also do this
   server-side if you pass `dimensions` in the request — same maths, and it
   still costs you a call. Truncating what you already have is free.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import env as _env

CACHE_DIR = _env.ROOT / ".cache"

# Azure caps a single embeddings request. Well under the limit and large
# enough that the corpus is a couple of round trips.
BATCH = 64


class AzureEmbedError(RuntimeError):
    pass


def _post(url: str, key: str, body: dict, timeout: int = 60) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("api-key", key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:800]
        raise AzureEmbedError("HTTP {} from {}\n{}".format(e.code, url, detail))
    except urllib.error.URLError as e:
        raise AzureEmbedError("Could not reach {}\n  {}".format(url, e.reason))


class AzureEmbeddingModel:
    """
    Same surface as indexing.EmbeddingModel: .name, .dims,
    .price_per_1m_tokens, .embed(text, dims). Nothing above this knows the
    difference.
    """

    def __init__(self, name, deployment, dims, price_per_1m_tokens,
                 endpoint, api_key, api_version=None):
        self.name = name
        self.deployment = deployment
        self.dims = int(dims)
        self.price_per_1m_tokens = float(price_per_1m_tokens)
        self.endpoint = _env.normalise_aoai_endpoint(endpoint)
        self.api_key = api_key
        self.api_version = api_version or None       # None => v1 route
        self.tokens_used = 0
        self.api_calls = 0
        self._mem = {}
        CACHE_DIR.mkdir(exist_ok=True)
        self._cache_path = CACHE_DIR / "{}.jsonl".format(
            _slug("{}|{}|{}".format(self.endpoint, self.deployment, self.dims)))
        self._load_cache()

    # ---------------- the two routes ----------------
    def _url(self) -> str:
        if self.api_version:
            return "{}/openai/deployments/{}/embeddings?api-version={}".format(
                self.endpoint, self.deployment, self.api_version)
        return "{}/openai/v1/embeddings".format(self.endpoint)

    def _body(self, inputs) -> dict:
        body = {"input": inputs}
        if not self.api_version:
            # v1 route names the deployment in the body, not the path.
            body["model"] = self.deployment
        return body

    def route_name(self) -> str:
        return "classic (api-version={})".format(self.api_version) if self.api_version else "v1"

    # ---------------- cache ----------------
    def _load_cache(self):
        if not self._cache_path.exists():
            return
        for line in self._cache_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                self._mem[rec["k"]] = rec["v"]
            except (ValueError, KeyError):
                continue        # a truncated last line is not worth dying over

    def _append_cache(self, key, vec):
        with open(self._cache_path, "a") as f:
            f.write(json.dumps({"k": key, "v": vec}) + "\n")

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    # ---------------- the call ----------------
    def embed_many(self, texts, progress=None):
        """Embed a list of texts at FULL dimensions, using and filling the
        cache. Returns vectors in the order given."""
        want = []
        for t in texts:
            k = self._key(t)
            if k not in self._mem:
                want.append(t)
        # dedupe while preserving order
        seen, todo = set(), []
        for t in want:
            k = self._key(t)
            if k not in seen:
                seen.add(k)
                todo.append(t)

        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            resp = self._request_with_retry(batch)
            self.api_calls += 1
            self.tokens_used += resp.get("usage", {}).get("prompt_tokens", 0)
            for item in resp["data"]:
                text = batch[item["index"]]
                k = self._key(text)
                vec = item["embedding"]
                self._mem[k] = vec
                self._append_cache(k, vec)
            if progress:
                progress(min(i + BATCH, len(todo)), len(todo))

        return [self._mem[self._key(t)] for t in texts]

    def _request_with_retry(self, batch, attempts: int = 4) -> dict:
        """429 is not an error, it is the service telling you the rate. Back
        off and try again — and if it never clears, say so plainly."""
        delay = 2.0
        last = None
        for n in range(attempts):
            try:
                return _post(self._url(), self.api_key, self._body(batch))
            except AzureEmbedError as e:
                last = e
                msg = str(e)
                if "HTTP 429" in msg or "HTTP 503" in msg or "HTTP 500" in msg:
                    if n < attempts - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                raise
        raise last

    def embed(self, text: str, dims=None):
        """One text. Truncate + renormalise from the cached full vector."""
        vec = self.embed_many([text])[0]
        return _truncate(vec, dims or self.dims)

    def embed_all(self, texts, dims=None, progress=None):
        full = self.embed_many(texts, progress=progress)
        d = dims or self.dims
        return [_truncate(v, d) for v in full]

    @property
    def cost_usd(self) -> float:
        return self.tokens_used / 1_000_000.0 * self.price_per_1m_tokens


def _truncate(vec, d: int):
    """Matryoshka truncation: keep the first d components, renormalise.

    Renormalising matters. Cosine similarity on unnormalised truncated
    vectors is not the same ranking, and the bug is invisible — the numbers
    still look like scores."""
    if d >= len(vec):
        return vec
    head = vec[:d]
    n = math.sqrt(sum(x * x for x in head)) or 1e-9
    return [x / n for x in head]


def _slug(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ==========================================================================
# PROBE — answer "which route does this resource serve, and what is deployed"
# before you spend an afternoon on a 404.
# ==========================================================================
def probe(endpoint: str, api_key: str, deployment: str, api_version=None):
    """Try the configured route, then the other one. Report what happened."""
    ep = _env.normalise_aoai_endpoint(endpoint)
    results = []
    routes = [("v1", None), ("classic", api_version or "2024-12-01-preview")]
    if api_version:
        routes.reverse()          # try what they configured first
    for label, ver in routes:
        m = AzureEmbeddingModel("probe", deployment, 8, 0.0, ep, api_key, ver)
        try:
            t0 = time.time()
            resp = _post(m._url(), api_key, m._body(["ping"]))
            ms = (time.time() - t0) * 1000
            n = len(resp["data"][0]["embedding"])
            results.append((label, ver, True,
                            "OK  {} dims, {:.0f} ms".format(n, ms)))
        except AzureEmbedError as e:
            first = str(e).splitlines()[0]
            results.append((label, ver, False, first))
    return ep, results


def build_models(cfg) -> dict:
    """Construct the small/large pair from resolved config."""
    out = {}
    for lever in ("small", "large"):
        up = lever.upper()
        dep = _env.get("AZURE_EMBED_DEPLOYMENT_" + up, required=True,
                       hint="The name you gave the text-embedding-3-{} "
                            "deployment in the portal.".format(lever))
        out[lever] = AzureEmbeddingModel(
            name=lever,
            deployment=dep,
            dims=_env.get_int("AZURE_EMBED_DIMS_" + up,
                              1536 if lever == "small" else 3072),
            price_per_1m_tokens=_env.get_float("AZURE_EMBED_PRICE_" + up,
                                               0.02 if lever == "small" else 0.13),
            endpoint=cfg["endpoint"],
            api_key=cfg["api_key"],
            api_version=cfg["api_version"],
        )
    return out


def resolve_aoai() -> dict:
    return {
        "endpoint": _env.normalise_aoai_endpoint(
            _env.get("AZURE_OPENAI_ENDPOINT", required=True,
                     hint="Portal > your Azure OpenAI resource > "
                          "Keys and Endpoint > Endpoint.")),
        "api_key": _env.get("AZURE_OPENAI_API_KEY", required=True,
                            hint="Portal > Keys and Endpoint > KEY 1."),
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION") or None,
    }
