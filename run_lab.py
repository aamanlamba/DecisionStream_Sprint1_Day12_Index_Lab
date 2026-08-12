"""
run_lab.py — build the index, run the queries, measure what filtering buys.

OFFLINE (default — no Azure, no key, no pip install)
    python3 run_lab.py                     # the full grid
    python3 run_lab.py --filters none      # what happens with no metadata
    python3 run_lab.py --show Q01          # one query, every configuration
    python3 run_lab.py --dims              # dimension truncation experiment

LIVE — LAYER 1: real Azure OpenAI embeddings, the lab's own filter index
    python3 run_lab.py --probe             # which route does my resource serve?
    python3 run_lab.py --live embed        # swap ONE thing: the embeddings
    python3 run_lab.py --live embed --dims # real Matryoshka truncation

LIVE — LAYER 2: a real Azure AI Search index
    python3 run_lab.py --live search --provision   # create index + upload
    python3 run_lab.py --live search               # query it
    python3 run_lab.py --live search --filter-mode strictPostFilter
    python3 run_lab.py --live search --teardown    # delete the index

Writes to out/:
    index_report.md     the report for your ADR
    results.csv         every configuration
    per_query.csv       where each configuration wins and loses
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "index_lab"))

from corpus import DOCS, QUERIES                                # noqa: E402
from indexing import build_index, chunk_docs, CHUNKERS, MODELS  # noqa: E402
import env as envlib                                            # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

# The three filter strategies. This is the axis that matters today.
FILTER_MODES = {
    "none":      lambda q: None,
    "jurisdiction": lambda q: {"jurisdiction": q["jurisdiction"]},
    "full":      lambda q: {"jurisdiction": q["jurisdiction"],
                            "policy_type": q.get("policy_type"),
                            "in_force_on": q["decision_date"]},
}

# --------------------------------------------------------------------------
# THE NEGATIVE-QUERY THRESHOLD, AND WHY IT IS NOT ONE NUMBER
#
# Q10 asks for something the corpus does not contain. "Correct" means the top
# hit is WEAK — the index admits it has nothing. To decide what weak means you
# need a cutoff, and here is the thing nobody tells you until it bites:
#
#   THE THREE RETRIEVAL MODES RETURN THREE INCOMMENSURABLE SCORE SCALES.
#
#     BM25     unbounded. 8.0 is a good match; there is no maximum.
#     vector   cosine, mapped into roughly 0..1 by the service.
#     hybrid   Reciprocal Rank Fusion. Bounded near 1/60 per result set, so
#              a PERFECT hybrid hit scores about 0.03.
#
# A single "score > 0.7 means relevant" rule applied across these is not a
# strict threshold — it is three different rules, two of which are nonsense.
# This is a real production failure: teams set a relevance floor against
# vector search, later enable hybrid, and every result drops below the floor.
#
# So the cutoff is per backend AND per mode, it is CONFIGURATION rather than a
# constant, and `--calibrate` measures the separation on your own service
# instead of trusting the numbers below.
# --------------------------------------------------------------------------
# A FOURTH SCALE, AND THE ONE THAT CAUGHT THIS LAB OUT
#
# The simulated embeddings are hashed into a high-dimensional space, so two
# unrelated chunks come out very nearly ORTHOGONAL: across the corpus the
# median cosine to a query is 0.000 and the floor is 0.000. Real
# text-embedding-3 vectors do not behave like that at all. Every piece of
# English is somewhat similar to every other piece of English, so the same
# measurement gives a median of 0.274 and a FLOOR of 0.153.
#
# So 0.25 — a sensible "this is basically nothing" cutoff against the
# simulated vectors — sits BELOW the real model's noise floor, and Q10 goes
# from passing everywhere to failing everywhere. That is not the retrieval
# getting worse. It is a constant that was only ever valid for the thing it
# was tuned against, silently carried onto something else.
#
# It is the same mistake as shipping a vector-tuned relevance floor and then
# enabling hybrid. Measure it with --calibrate; do not inherit it.
NEGATIVE_THRESHOLD = {
    # Offline: simulated vectors, near-orthogonal, scores hug zero.
    ("offline", "keyword"): 0.25,
    ("offline", "vector"): 0.25,
    ("offline", "hybrid"): 0.25,
    # Layer 1: the lab's own scoring formula, but over REAL embeddings.
    # Measured with `--live embed --calibrate` against text-embedding-3-small
    # and -large on 2026-08-12. Re-measure on your own resource.
    # keyword stays at 0.25 on purpose: BM25-style term overlap never touches
    # the embeddings, so its scores are byte-identical to the offline run.
    # Only the modes that consume a vector move.
    ("live_embed", "keyword"): envlib.get_float("DS_NEG_EMBED_KEYWORD", 0.25),
    ("live_embed", "vector"): envlib.get_float("DS_NEG_EMBED_VECTOR", 0.49),
    ("live_embed", "hybrid"): envlib.get_float("DS_NEG_EMBED_HYBRID", 0.34),
    # Layer 2: Azure's own scoring — BM25, mapped cosine, RRF. Measured with
    # `--live search --calibrate` on 2026-08-12 against a 45-chunk index.
    #
    # LOOK AT HOW LITTLE ROOM THERE IS IN THE VECTOR ROW. A query the corpus
    # can answer scored 0.655; a query it cannot answer scored 0.593. Sixty-two
    # thousandths apart. Any "is this relevant enough" rule you build on a
    # single cosine is balanced on that margin, and it will not survive a
    # corpus change. BM25 separated the same pair by 3.50 — because a document
    # that shares no terms with the query scores exactly zero, and a vector
    # never does.
    #
    # If you need a reliable "we do not have this" signal, do not get it from
    # a cosine threshold.
    ("azure", "keyword"): envlib.get_float("DS_NEG_BM25", 1.75),
    ("azure", "vector"): envlib.get_float("DS_NEG_VECTOR", 0.62),
    ("azure", "hybrid"): envlib.get_float("DS_NEG_HYBRID", 0.025),
}


def judge(hits, query, backend="offline", mode="hybrid"):
    """Correct means: the right document, containing the right wording, at
    rank 1. Anything else is a wrong answer that will read perfectly well."""
    if query["expect_doc"] is None:
        # Negative query. Correct behaviour is a weak top hit.
        if not hits:
            return True, "no candidates"
        top, score = hits[0]
        cut = NEGATIVE_THRESHOLD[(backend, mode)]
        return (score < cut), "top score {:.3f} (cut {:.3f})".format(score, cut)
    if not hits:
        return False, "nothing retrieved"
    top, score = hits[0]
    if top.doc_id != query["expect_doc"]:
        return False, "wrong doc: {}".format(top.doc_id)
    if query["expect_text"] and query["expect_text"].lower() not in top.text.lower():
        return False, "right doc, wrong chunk ({})".format(top.id)
    return True, "ok"


# ==========================================================================
# OFFLINE / LAYER-1 GRID
# ==========================================================================
def run_config(chunker, model, fmode, top_k, dims=None, mode="hybrid",
               models=None, backend="offline"):
    idx = build_index(DOCS, chunker, model, dims, models=models)
    registry = models if models is not None else MODELS
    rows, correct = [], 0
    for q in QUERIES:
        flt = FILTER_MODES[fmode](q)
        hits = idx.search(q["q"], top_k, flt, mode=mode)
        ok, why = judge(hits, q, backend, mode)
        correct += ok
        rows.append({"qid": q["id"], "type": q["type"], "correct": ok, "why": why,
                     "candidates": idx.candidate_count(flt),
                     "top": hits[0][0].doc_id if hits else "-"})
    return {
        "chunker": chunker, "model": model, "filters": fmode, "retrieval": mode,
        "dims": dims or registry[model].dims,
        "chunks": len(idx.chunks), "correct": correct, "of": len(QUERIES),
        "avg_candidates": round(sum(r["candidates"] for r in rows) / len(rows), 1),
        "embed_tokens": idx.embed_tokens,
        "embed_cost": round(idx.embed_tokens / 1_000_000
                            * registry[model].price_per_1m_tokens, 6),
        "rows": rows,
    }


def show(qid, top_k, models=None, backend="offline"):
    q = next((x for x in QUERIES if x["id"] == qid), None)
    if not q:
        print("No such query. Try: {}".format(", ".join(x["id"] for x in QUERIES)))
        return 1
    print("\n{}  [{}]  {}".format(q["id"], q["type"], q["q"]))
    print("  context: jurisdiction={}  decision_date={}  policy_type={}".format(
        q["jurisdiction"], q["decision_date"], q.get("policy_type")))
    print("  TRAP: {}".format(q["trap"]))
    print("  expected: {}  containing '{}'\n".format(q["expect_doc"], q["expect_text"]))
    for model in ("small", "large"):
        for fmode in FILTER_MODES:
            idx = build_index(DOCS, "semantic", model, models=models)
            flt = FILTER_MODES[fmode](q)
            hits = idx.search(q["q"], top_k, flt, mode="hybrid")
            ok, why = judge(hits, q, backend, "hybrid")
            tops = ", ".join("{}({:.2f})".format(c.doc_id, s) for c, s in hits[:3])
            print("  {:<6} filters={:<13} {} cands={:<3} {}".format(
                model, fmode, "OK " if ok else "x  ", idx.candidate_count(flt), tops))
            if not ok:
                print("         -> {}".format(why))
    return 0


def calibrate_index(models, top_k, backend):
    """
    --calibrate for Layer 1: what do the scores actually look like?

    Same job as the Azure version, one layer down. It also prints the score
    FLOOR across the whole corpus, because that is the number that explains
    why a threshold does not transfer: the simulated vectors bottom out at
    0.00 and the real ones bottom out around 0.15.
    """
    pos, neg = QUERIES[0], QUERIES[-1]
    label = "REAL Azure OpenAI" if models else "simulated"
    print("\nCALIBRATION — negative-query cutoff ({} embeddings)".format(label))
    print("  a real answer   : {}".format(pos["q"]))
    print("  nothing to find : {}\n".format(neg["q"]))
    print("  {:<7} {:<9} {:>10} {:>11} {:>11} {:>13}".format(
        "model", "mode", "real top", "absent top", "corpus floor", "suggested cut"))
    print("  " + "-" * 68)
    suggestions = {}
    for lever in ("small", "large"):
        idx = build_index(DOCS, "semantic", lever, models=models)
        for mode in ("keyword", "vector", "hybrid"):
            f = FILTER_MODES["full"]
            hp = idx.search(pos["q"], top_k, f(pos), mode=mode)
            hn = idx.search(neg["q"], top_k, f(neg), mode=mode)
            allh = idx.search(pos["q"], 10 ** 6, None, mode=mode)
            floor = min([s for _, s in allh]) if allh else 0.0
            sp = hp[0][1] if hp else 0.0
            sn = hn[0][1] if hn else 0.0
            cut = round((sp + sn) / 2, 2) if sp > sn else round(sn * 1.05, 2)
            suggestions.setdefault(mode, []).append(cut)
            print("  {:<7} {:<9} {:>10.3f} {:>11.3f} {:>12.3f} {:>13.2f}".format(
                lever, mode, sp, sn, floor, cut))
    print("\n  The 'corpus floor' column is the point. Two chunks of unrelated")
    print("  English are not orthogonal to a real embedding model, so its")
    print("  scores never approach zero. A cutoff tuned against the simulated")
    print("  vectors sits below that floor and marks every result 'strong'.")
    print("\n  Set these in .env for --live embed:")
    for mode in ("keyword", "vector", "hybrid"):
        print("    DS_NEG_EMBED_{:<8}={}".format(
            mode.upper(), max(suggestions[mode])))
    return 0


def dims_experiment(top_k, models=None, backend="offline"):
    """
    Truncation is only interesting where the embedding is doing work.

    Run it BOTH ways: with tight filters and with none. The difference
    between the two columns is the finding.
    """
    registry = models if models is not None else MODELS
    native = registry["large"].dims
    print("\nDIMENSION TRUNCATION — large model, semantic chunking")
    if models is not None:
        print("(real text-embedding-3 vectors, native {} dims, truncated "
              "client-side and renormalised)".format(native))
    print("{:>6} {:>14} {:>14} {:>14} {:>10}".format(
        "dims", "filters=full", "filters=none", "index floats", "rel size"))
    print("-" * 64)
    ladder = [d for d in (3072, 1536, 1024, 768, 512, 384, 256, 128, 64, 32)
              if d <= native]
    base = None
    for d in ladder:
        rf = run_config("semantic", "large", "full", top_k, dims=d, mode="vector",
                        models=models, backend=backend)
        rn = run_config("semantic", "large", "none", top_k, dims=d, mode="vector",
                        models=models, backend=backend)
        floats = rf["chunks"] * d
        base = base or floats
        print("{:>6} {:>11}/{:<2} {:>11}/{:<2} {:>14,} {:>9.0%}".format(
            d, rf["correct"], rf["of"], rn["correct"], rn["of"], floats, floats / base))
    print("\nRead the two columns against each other.")
    print("With tight filters the candidate set is small, so the embedding has")
    print("very little discriminating to do and truncation costs you almost")
    print("nothing. With no filters it is doing all the work, and shrinking it")
    print("shows. Dimension reduction is a cheap saving IF your filters are good.")
    print("It is a false economy if they are not.")
    return 0


# ==========================================================================
# REPORT
# ==========================================================================
CHUNKER_DESC = {
    "fixed_256": "Fixed windows of **256 tokens** (~1,024 characters) with "
                 "~128 characters of overlap. Cuts wherever the window ends — "
                 "including halfway through a clause.",
    "fixed_512": "Fixed windows of **512 tokens** (~2,048 characters) with "
                 "~256 characters of overlap. Same method, bigger window.",
    "semantic":  "Split on the document's **own structure** — `CLAUSE 4.1`, "
                 "`SECTION 2` headings. A clause stays whole and its heading "
                 "travels with it, so the chunk still says what it is.",
}

FILTER_DESC = {
    "none": ("No metadata at all. Every chunk in the corpus is a candidate. "
             "This is what you get if you index the text and nothing else."),
    "jurisdiction": ("Filter on country only — `jurisdiction eq 'UK'`. "
                     "The obvious half-measure."),
    "full": ("Country **and** product **and** in-force-on-the-decision-date: "
             "`jurisdiction` + `policy_type` + `effective_from`/`effective_to` "
             "spanning the date the claim was decided."),
}

RETRIEVAL_DESC = {
    "offline": {
        "keyword": "Term overlap between query and chunk, weighted by IDF so "
                   "rare words count for more. Blind to meaning; excellent at "
                   "clause numbers.",
        "vector":  "Cosine similarity between the query embedding and each "
                   "chunk embedding. Blind to identifiers like `3.5`.",
        "hybrid":  "A fixed 50/50 blend of the two scores above.",
    },
    "azure": {
        "keyword": "Real **BM25** full-text search over the `content` field.",
        "vector":  "Real **HNSW** approximate-nearest-neighbour search over "
                   "cosine distance — run exhaustively here, because on "
                   "{n_chunks} chunks the approximation has nothing to "
                   "approximate and exhaustive keeps the lab reproducible. In "
                   "production you want the approximation; it is what you are "
                   "paying HNSW for.",
        "hybrid":  "Both, fused by **Reciprocal Rank Fusion (RRF)** — Azure "
                   "combines the two ranked lists by position, not by score.",
    },
}


def _matched_pair(results, key, a, b):
    """Find two rows identical except for `key`, to use as a worked example."""
    def sig(r):
        return tuple(r.get(k) for k in
                     ("chunker", "model", "retrieval", "filter_mode") if k != key)
    left = dict((sig(r), r) for r in results if r.get(key) == a)
    for r in results:
        if r.get(key) == b and sig(r) in left:
            return left[sig(r)], r
    return None, None


def n_chunks_for_note(results):
    return max([r["chunks"] for r in results] or [0])


def glossary(results, meta):
    """Define every term in the tables above, with this run's real numbers.

    A report whose column headings need explaining in person is a report that
    will be misread the moment you leave the room."""
    backend = "azure" if meta.get("live_search") else "offline"
    chunk_counts = {}
    dims = {}
    for r in results:
        chunk_counts.setdefault(r["chunker"], r["chunks"])
        dims.setdefault(r["model"], r["dims"])

    md = ["## How to read this report", "",
          "**One row in the results table is one complete configuration of the "
          "retrieval pipeline.** Each column is an independent choice you would "
          "have to make for real, and the run varies them one at a time so you "
          "can attribute a change to a cause.", "",
          "### Chunker — how a document is cut up before it is embedded", "",
          "You do not embed whole documents. You cut them into chunks, embed "
          "each chunk, and retrieve chunks. Where you cut decides what a "
          "retrieved answer contains.", "",
          "| Setting | What it does | Chunks from {} docs |".format(len(DOCS)),
          "|---|---|---|"]
    for c in sorted(chunk_counts):
        md.append("| `{}` | {} | {} |".format(
            c, CHUNKER_DESC.get(c, "—"), chunk_counts[c]))

    if "fixed_256" in chunk_counts and "fixed_512" in chunk_counts:
        md += ["",
               "> **Read the chunk counts before you read the chunker result.** "
               "`fixed_256` and `fixed_512` produce {} and {} chunks here — "
               "almost identical, because most of these policy documents are "
               "shorter than a single 1,024-character window and are never "
               "split at all. On this corpus the two fixed settings are very "
               "nearly the *same experiment*, so the small chunker swing below "
               "is partly a property of the sample, not a finding about chunk "
               "size. On real policy PDFs of 40 pages it would separate them."
               .format(chunk_counts["fixed_256"], chunk_counts["fixed_512"])]

    md += ["", "### Embedding — which model turns text into a vector", "",
           "| Setting | What it is |", "|---|---|"]
    if meta.get("model_desc"):
        for lever, desc in meta["model_desc"].items():
            md.append("| `{}` | {} |".format(lever, desc))
    else:
        for m in sorted(dims):
            md.append("| `{}` | {} dimensions |".format(m, dims[m]))

    md += ["", "### Filters — which metadata is applied before the search runs",
           "", "**This is the axis the lab is actually testing.** The filter "
           "decides which chunks are even eligible to be compared against the "
           "query. It is applied *before* similarity, not after.", "",
           "| Setting | What it does |", "|---|---|"]
    for f in ("none", "jurisdiction", "full"):
        if any(r["filters"] == f for r in results):
            md.append("| `{}` | {} |".format(f, FILTER_DESC[f]))

    md += ["", "### Retrieval — how candidates are scored once filtered", "",
           "| Setting | What it does |", "|---|---|"]
    n_chunks = max([r["chunks"] for r in results] or [0])
    for rmode in ("keyword", "vector", "hybrid"):
        if any(r["retrieval"] == rmode for r in results):
            md.append("| `{}` | {} |".format(
                rmode, RETRIEVAL_DESC[backend][rmode].replace(
                    "{n_chunks}", str(n_chunks))))

    if any("filter_mode" in r for r in results):
        md += ["", "### vectorFilterMode — *when* the filter is applied", "",
               "| Setting | What it does | Risk |", "|---|---|---|",
               "| `preFilter` | Applies the filter **during** HNSW traversal on "
               "each shard, expanding the graph until k candidates are found. "
               "Default since ~Oct 2023, and recommended. | None — guarantees "
               "k results if they exist |",
               "| `postFilter` | Traverses each shard **without** the filter to "
               "get that shard's local top-k, filters *that*, then aggregates. "
               "| Moderate — misses matches per shard |",
               "| `strictPostFilter` | Finds the **unfiltered global top-k** "
               "first, then filters it. Preview; needs a preview api-version. "
               "| Highest — can return **zero** results when matches exist |", "",
               "> **`postFilter` is a no-op on a corpus this small, and that is "
               "not a bug.** It degrades *per shard*, and {} chunks is one shard "
               "whose local top-k is effectively the whole set. Microsoft's own "
               "benchmark calls the two modes \"approximately equal\" on small "
               "indexes; the divergence needs 10^5–10^6 documents across several "
               "shards. `strictPostFilter` filters the *global* top-k, so it "
               "fails at any size — which is why it is the one that demonstrates "
               "the point here. Do not generalise its result into a claim about "
               "`postFilter`.".format(n_chunks_for_note(results))]

    md += ["", "### The columns", "",
           "| Column | What it means |", "|---|---|",
           "| **Correct** | Out of {} queries. Strict: the **top-ranked** hit "
           "must be the expected document *and* contain the expected wording. "
           "A right answer at rank 2 scores zero, because a RAG pipeline that "
           "passes the top hit to a model never sees rank 2. |".format(len(QUERIES)),
           "| **Avg candidates** | How many chunks survived the filter, "
           "averaged over the {} queries. **This is the column that explains "
           "the Correct column** — it is the size of the haystack the search "
           "was asked to work in. |".format(len(QUERIES)),
           "| **Index cost** | One-off USD to embed the whole corpus once at "
           "that chunker + embedding combination. Not a per-query cost — a "
           "per-*rebuild* cost, which is what you pay every time the metadata "
           "schema changes. |", ""]

    # ---- worked example, with this run's real numbers ----
    lo, hi = _matched_pair(results, "filters", "none", "full")
    if lo and hi:
        md += ["### Worked example — reading one pair of rows", "",
               "```",
               "chunker={:<9} embedding={:<6} retrieval={:<8} filters=none  "
               "-> {}/{} correct, {} candidates".format(
                   lo["chunker"], lo["model"], lo["retrieval"],
                   lo["correct"], lo["of"], lo["avg_candidates"]),
               "chunker={:<9} embedding={:<6} retrieval={:<8} filters=full  "
               "-> {}/{} correct, {} candidates".format(
                   hi["chunker"], hi["model"], hi["retrieval"],
                   hi["correct"], hi["of"], hi["avg_candidates"]),
               "```", "",
               "Same chunker. Same embedding model. Same retrieval method. "
               "The **only** difference is the metadata filter, and it is "
               "worth **{} queries out of {}**.".format(
                   hi["correct"] - lo["correct"], hi["of"]), "",
               "The candidate count is why. It fell from about {:.0f} chunks "
               "to about {:.0f}: the search stopped having to tell the right "
               "clause apart from roughly {:.0f} other well-written "
               "candidates, because those were never eligible to be compared "
               "in the first place. Nothing about the search got smarter — "
               "the haystack got smaller.".format(
                   lo["avg_candidates"], hi["avg_candidates"],
                   max(0, lo["avg_candidates"] - hi["avg_candidates"])), ""]

    md += ["### What 'swing' means, and why it is the number to read", "",
           "**Swing** = best average minus worst average for that lever, "
           "holding the whole rest of the grid constant. It answers the only "
           "question that matters when you are deciding where to spend a "
           "sprint: *if I change this one thing, how much does the answer "
           "improve?*", "",
           "A lever with a big swing is worth your time. A lever with a swing "
           "under 1 is a preference, not an engineering decision — and "
           "arguing about it in a design review is how a sprint disappears.", "",
           "### What this report does NOT tell you", "",
           "- **Nothing about answer quality.** It measures retrieval only — "
           "whether the right chunk reaches rank 1. What a model then writes "
           "with that chunk is a different evaluation.",
           "- **Nothing about latency or scale.** {} documents is a teaching "
           "corpus. HNSW, quantisation and shard counts start to matter three "
           "orders of magnitude further up.".format(len(DOCS)),
           "- **Nothing about recall beyond rank 1.** A configuration scoring "
           "3/10 here might have the right chunk at rank 2 every time. That "
           "would be a different lab, and a kinder one.", ""]
    return md


def write_report(results, top_k, meta=None):
    meta = meta or {}
    cols = ["chunker", "model", "filters", "retrieval", "dims", "chunks", "correct", "of",
            "avg_candidates", "embed_tokens", "embed_cost"]
    extra = [c for c in ("filter_mode",) if any(c in r for r in results)]
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + extra)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in cols + extra})

    with open(OUT / "per_query.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunker", "model", "filters", "retrieval", "qid", "type", "correct",
                    "top_doc", "why", "candidates"])
        for r in results:
            for q in r["rows"]:
                w.writerow([r["chunker"], r["model"], r["filters"], r["retrieval"], q["qid"],
                            q["type"], q["correct"], q["top"], q["why"], q["candidates"]])

    # Lever averages use preFilter rows only — see paired_filter_mode() for
    # why mixing the two vectorFilterModes into one average is misleading.
    base = [r for r in results if r.get("filter_mode", "preFilter") == "preFilter"]

    def agg(key):
        out = {}
        for r in base:
            out.setdefault(r[key], []).append(r["correct"])
        return dict((k, round(sum(v) / len(v), 2)) for k, v in out.items())

    best = max(base, key=lambda r: (r["correct"], -r["embed_cost"]))
    title = meta.get("title", "Azure AI Search index report — DecisionStream AI")
    md = ["# " + title, ""]
    if meta.get("banner"):
        md += [meta["banner"], ""]
    md += ["Documents: **{}**   Queries: **{}**   top_k: **{}**".format(
        len(DOCS), len(QUERIES), top_k), "",
        "*Read the glossary first if any column heading is unfamiliar — the "
        "results are easy to misread without it.*", ""]
    md += glossary(results, meta)
    has_vfm = any("filter_mode" in r for r in results)
    md += ["## All configurations", "",
           "| Chunker | Embedding | Filters |{} Retrieval | Correct | "
           "Avg candidates | Index cost |".format(" vectorFilterMode |" if has_vfm else ""),
           "|---|---|---|{}---|---|---|---|".format("---|" if has_vfm else "")]
    for r in sorted(results, key=lambda x: -x["correct"]):
        md.append("| {} | {} | {} |{} {} | {}/{} | {} | ${:.6f} |".format(
            r["chunker"], r["model"], r["filters"],
            " {} |".format(r["filter_mode"]) if has_vfm else "",
            r["retrieval"], r["correct"], r["of"], r["avg_candidates"],
            r["embed_cost"]))

    md += ["", "## Which lever actually moved the result", "",
           "Average correctness, holding everything else across the grid:", "",
           "| Lever | Setting | Avg correct |", "|---|---|---|"]
    levers = [l for l in ("filters", "retrieval", "model", "chunker")
              if any(l in r for r in base) and len(agg(l)) > 1]
    for lever in levers:
        for k, v in sorted(agg(lever).items(), key=lambda x: -x[1]):
            md.append("| {} | {} | {} |".format(lever, k, v))

    md += ["", "### Read the swing, not the winner", ""]
    for lever in levers:
        a = agg(lever)
        md.append("- Changing the **{}** moves correctness by **{:.2f}** queries".format(
            lever, max(a.values()) - min(a.values())))
    md += ["",
           "Metadata filtering is not a tuning parameter. It is the difference "
           "between answering about the right country and the right year, and "
           "answering fluently about the wrong one. No embedding model closes "
           "that gap, because it is not a similarity problem.", "",
           "## Best configuration", "",
           "**{} + {} + filters={}** — {}/{} correct, {} chunks, "
           "index cost ${:.6f}.".format(
               best["chunker"], best["model"], best["filters"], best["correct"],
               best["of"], best["chunks"], best["embed_cost"]), ""]

    pairs = paired_filter_mode(results)
    if pairs:
        md += ["## vectorFilterMode, paired against preFilter", "",
               "Only configurations where both modes ran. The post-filter modes "
               "are skipped for keyword search and for `filters=none`, where "
               "they mean nothing — so an unpaired average would compare "
               "different subsets and flatter them by omitting the rows they "
               "break.", "",
               "| Filters | Retrieval | Mode | preFilter | That mode | Delta |",
               "|---|---|---|---|---|---|"]
        for (ch, mo, fm, rm), mode, a, b in pairs:
            md.append("| {} | {} | `{}` | {}/{} | {}/{} | {:+d} |".format(
                fm, rm, mode, a, len(QUERIES), b, len(QUERIES), b - a))

        deltas = {}
        for _, mode, a, b in pairs:
            deltas.setdefault(mode, []).append(b - a)
        md += [""]
        if "postFilter" in deltas and all(d == 0 for d in deltas["postFilter"]):
            md += ["**`postFilter` shows no difference here, and that is the "
                   "honest result rather than a demonstration.** It degrades "
                   "*per shard*; {} chunks is a single shard whose local top-k "
                   "is effectively the whole set. This lab cannot measure that "
                   "mode — it can only explain it.".format(
                       n_chunks_for_note(results)), ""]
        if deltas.get("strictPostFilter") and min(deltas["strictPostFilter"]) < 0:
            md += ["**`strictPostFilter` does fail here, by up to {} queries.** "
                   "It filters the *global* top-k, so it breaks at any corpus "
                   "size: the k nearest across the whole index are the wrong "
                   "jurisdiction, they are discarded, and the right clause — "
                   "which ranked below them — was never in the running. You ask "
                   "for 3 results and get 0.".format(
                       -min(deltas["strictPostFilter"])), "",
                   "The filter is not a post-processing step. It decides what "
                   "is eligible to be compared at all.", ""]

    if meta.get("notes"):
        md += ["## What this run actually used", ""] + meta["notes"] + [""]

    md += ["## Questions to answer before this goes in the ADR", "",
           "1. Which metadata fields are REQUIRED at ingestion, and what happens "
           "to a document that arrives without them?",
           "2. Q06 asks about a claim decided in June 2025 and the current policy "
           "is the wrong answer. How does your index know the decision date?",
           "3. Your corpus contains superseded versions. Should they be indexed "
           "at all — and what breaks if you delete them?",
           "4. What does the embedding cost look like on a full re-index, and how "
           "often will you re-index?",
           "5. Which of your ten queries would still fail, and who finds out?",
           "6. What is your review trigger for rebuilding this index?", ""]

    (OUT / "index_report.md").write_text("\n".join(md))


def paired_filter_mode(results):
    """
    pre- versus postFilter, compared ONLY where both actually ran.

    This has to be paired and it is worth knowing why. postFilter is skipped
    for keyword search and for filters=none, because it means nothing there.
    Averaging the two modes across the whole grid therefore compares
    different subsets — and it flatters postFilter, because the rows it
    misses are the ones where it does the damage. An unbalanced average is
    how a real regression hides inside a summary statistic.
    """
    pre, others = {}, {}
    for r in results:
        if "filter_mode" not in r:
            continue
        key = (r["chunker"], r["model"], r["filters"], r["retrieval"])
        if r["filter_mode"] == "preFilter":
            pre[key] = r["correct"]
        else:
            others.setdefault(r["filter_mode"], {})[key] = r["correct"]
    pairs = []
    for mode in ("postFilter", "strictPostFilter"):
        for key, val in others.get(mode, {}).items():
            if key in pre:
                pairs.append((key, mode, pre[key], val))
    pairs.sort(key=lambda x: (x[3] - x[2], x[1], x[0]))
    return pairs


def print_levers(results):
    def agg(key, rows):
        out = {}
        for r in rows:
            out.setdefault(r[key], []).append(r["correct"])
        return dict((k, sum(v) / len(v)) for k, v in out.items())

    # Lever averages are computed on preFilter rows only, so that adding the
    # postFilter experiment does not quietly move the other numbers.
    base = [r for r in results if r.get("filter_mode", "preFilter") == "preFilter"]

    print("-" * 76)
    print("AVERAGE CORRECTNESS BY LEVER")
    for lever in ("filters", "retrieval", "model", "chunker"):
        if not any(lever in r for r in base):
            continue
        a = agg(lever, base)
        if len(a) < 2:
            continue
        parts = "   ".join("{}={:.2f}".format(k, v)
                           for k, v in sorted(a.items(), key=lambda x: -x[1]))
        print("  {:<11} {}      swing {:.2f}".format(
            lever, parts, max(a.values()) - min(a.values())))
    print("\nThe lever with the biggest swing is the one worth your time.")

    pairs = paired_filter_mode(results)
    if pairs:
        print("\n" + "-" * 76)
        print("vectorFilterMode — each mode vs preFilter, PAIRED")
        print("  (only configurations where both modes ran; an unpaired average")
        print("   compares different subsets and hides exactly this)")
        print("\n  {:<14} {:<8} {:<17} {:>9} {:>9} {:>7}".format(
            "filters", "retr", "mode", "preFilter", "that mode", "delta"))
        print("  " + "-" * 70)
        for (ch, mo, fm, rm), mode, a, b in pairs:
            flag = "  <--" if b < a else ""
            print("  {:<14} {:<8} {:<17} {:>7}/10 {:>7}/10 {:>+7}{}".format(
                fm, rm, mode, a, b, b - a, flag))

        deltas = {}
        for _, mode, a, b in pairs:
            deltas.setdefault(mode, []).append(b - a)
        print("")
        if all(d == 0 for d in deltas.get("postFilter", [0])):
            print("  postFilter is INDISTINGUISHABLE from preFilter here, and that")
            print("  is not a bug. It degrades per SHARD — it filters each shard's")
            print("  local top-k — so showing it needs TWO things this lacks:")
            print("")
            print("    documents  {} chunks is not enough for a shard's local".format(
                max([r["chunks"] for r in results] or [0])))
            print("               top-k to be a real subset. Microsoft's benchmark")
            print("               puts the divergence at 10^5-10^6 documents.")
            print("    shards     shard count follows PARTITION count, which is a")
            print("               tier property: Basic ships with 1 (3 max) against")
            print("               12 on S1/S2/S3.")
            print("")
            print("  On a single-partition service this is not a 'small corpus'")
            print("  caveat you grow out of. Teach the mode; do not claim to have")
            print("  measured it. strictPostFilter is tier-independent — that is")
            print("  why it is the one that demonstrates the point.")
        if deltas.get("strictPostFilter"):
            worst = min(deltas["strictPostFilter"])
            if worst < 0:
                print("")
                print("  strictPostFilter DOES fail here, by up to {} queries. It".format(-worst))
                print("  filters the GLOBAL top-k, so it breaks at any corpus size:")
                print("  the k nearest across everything are the wrong jurisdiction,")
                print("  they are discarded, and the right clause — which sat below")
                print("  them — was never in the running. You ask for 3 and get 0.")
                print("")
                print("  That is the lesson. The filter is not a post-processing")
                print("  step; it decides what is eligible to be compared at all.")


def print_written():
    print("\nWritten: {}".format(OUT / "index_report.md"))
    print("         {}".format(OUT / "results.csv"))
    print("         {}".format(OUT / "per_query.csv"))


# ==========================================================================
# LIVE — LAYER 1 SETUP
# ==========================================================================
def azure_models():
    import azure_embed
    cfg = azure_embed.resolve_aoai()
    models = azure_embed.build_models(cfg)
    print("\nAZURE OPENAI EMBEDDINGS — live")
    print("  endpoint : {}".format(cfg["endpoint"]))
    print("  route    : {}".format(models["small"].route_name()))
    for lever in ("small", "large"):
        m = models[lever]
        print("  {:<8} : {} ({} dims, ${}/1M tokens)".format(
            lever, m.deployment, m.dims, m.price_per_1m_tokens))
    return models


def probe_cmd():
    import azure_embed
    cfg = azure_embed.resolve_aoai()
    print("\nPROBE — which data-plane route does this resource serve?")
    print("  endpoint: {}".format(cfg["endpoint"]))
    any_ok = False
    for lever in ("small", "large"):
        dep = os.environ.get("AZURE_EMBED_DEPLOYMENT_" + lever.upper())
        if not dep:
            print("\n  {}: AZURE_EMBED_DEPLOYMENT_{} not set — skipped".format(
                lever, lever.upper()))
            continue
        print("\n  deployment '{}':".format(dep))
        _, results = azure_embed.probe(cfg["endpoint"], cfg["api_key"], dep,
                                       cfg["api_version"])
        for label, ver, ok, msg in results:
            any_ok = any_ok or ok
            print("    {:<8} {} {}".format(label, "PASS" if ok else "FAIL", msg))
    print("\nIf ONE route passes, set AZURE_OPENAI_API_VERSION accordingly:")
    print("  v1 passed      -> comment the line out (leave it UNSET)")
    print("  classic passed -> set it to the api-version shown above")
    print("If BOTH fail on every deployment, the problem is the endpoint, the")
    print("key, or the deployment name — in that order of likelihood.")
    return 0 if any_ok else 1


# ==========================================================================
# LIVE — LAYER 2: AZURE AI SEARCH
# ==========================================================================
def search_backend(args, models):
    import azure_search
    cfg = azure_search.resolve_search()
    lever = envlib.get("AZURE_SEARCH_MODEL", "large")
    if lever not in models:
        lever = "large"
    embedder = models[lever]
    idx = azure_search.AzureSearchIndex(
        endpoint=cfg["endpoint"], api_key=cfg["api_key"],
        index_name=cfg["index"], api_version=cfg["api_version"],
        embedder=embedder, dims=embedder.dims,
        preview_api_version=cfg["preview_api_version"])
    print("\nAZURE AI SEARCH — live")
    print("  endpoint    : {}".format(idx.endpoint))
    print("  index       : {}".format(idx.name))
    print("  api-version : {}  (preview {} for strictPostFilter only)".format(
        idx.api_version, idx.preview_api_version))
    print("  vectors     : {} ({} dims, {} chunker)".format(
        embedder.deployment, idx.dims, args.chunker))
    note = idx.capacity_note()
    if note:
        print(note)
    return idx, cfg, lever


def provision(idx, chunker):
    def progress(done, total):
        print("\r    embedding {}/{}".format(done, total), end="", flush=True)

    chunks = chunk_docs(DOCS, chunker)
    print("\n  creating index '{}' ({} dims)...".format(idx.name, idx.dims))
    idx.create(recreate=True)
    print("  uploading {} chunks from {} documents...".format(len(chunks), len(DOCS)))
    n = idx.upload(chunks, progress=progress)
    print("\r    embedded {} chunks ({} API calls, {} tokens, ${:.6f})".format(
        len(chunks), idx.embedder.api_calls, idx.embedder.tokens_used,
        idx.embedder.cost_usd))
    print("  uploaded {} documents. waiting for them to become searchable...".format(n))
    live = idx.wait_until_indexed(n)
    if live < n:
        print("  WARNING: only {}/{} searchable after the wait. Indexing is "
              "asynchronous;\n           re-run the query step in a moment.".format(live, n))
    else:
        print("  {} documents searchable.".format(live))
    return live


def calibrate(idx, top_k):
    """Measure the score separation on THIS service, per retrieval mode.

    The point is not the numbers. It is that they do not live on a common
    scale, so a single relevance floor cannot be right for all three."""
    pos = QUERIES[0]        # Q01 — a query with a known good answer
    neg = QUERIES[-1]       # Q10 — the one that is not in the corpus
    print("\nCALIBRATION — negative-query cutoff, per retrieval mode")
    print("  a real answer   : {}".format(pos["q"]))
    print("  nothing to find : {}\n".format(neg["q"]))
    print("  {:<9} {:>12} {:>12} {:>12}   {}".format(
        "mode", "real top", "absent top", "separation", "suggested cut"))
    print("  " + "-" * 68)
    for mode in ("keyword", "vector", "hybrid"):
        f = FILTER_MODES["full"]
        hp = idx.search(pos["q"], top_k, f(pos), mode=mode)
        hn = idx.search(neg["q"], top_k, f(neg), mode=mode)
        sp = hp[0][1] if hp else 0.0
        sn = hn[0][1] if hn else 0.0
        cut = (sp + sn) / 2 if sp > sn else sn * 1.05
        print("  {:<9} {:>12.4f} {:>12.4f} {:>12.4f}   {:.4f}".format(
            mode, sp, sn, sp - sn, cut))
    print("\n  Three modes, three scales. BM25 is unbounded, vector is a mapped")
    print("  cosine, and hybrid is RRF which tops out near 0.03. A single")
    print("  relevance floor across all three is three different rules.")
    print("\n  Set the ones you want in .env:")
    print("    DS_NEG_BM25=   DS_NEG_VECTOR=   DS_NEG_HYBRID=")
    return 0


def run_search_grid(idx, args):
    results = []
    fmodes = [args.filters] if args.filters else list(FILTER_MODES)
    vmodes = ([args.filter_mode] if args.filter_mode
              else ["preFilter", "postFilter", "strictPostFilter"])
    modes = ["keyword", "vector", "hybrid"]
    lever = idx.embedder.name

    print("\n{:<14} {:<9} {:<17} {:>8} {:>6}".format(
        "filters", "retr", "vectorFilter", "correct", "cands"))
    print("-" * 76)
    for fm, rm, vm in itertools.product(fmodes, modes, vmodes):
        # vectorFilterMode is meaningless without a vector query or a filter.
        if vm != "preFilter" and (rm == "keyword" or fm == "none"):
            continue
        rows, correct = [], 0
        for q in QUERIES:
            flt = FILTER_MODES[fm](q)
            hits = idx.search(q["q"], args.top_k, flt, mode=rm,
                              vector_filter_mode=vm)
            ok, why = judge(hits, q, "azure", rm)
            correct += ok
            rows.append({"qid": q["id"], "type": q["type"], "correct": ok,
                         "why": why, "candidates": idx.candidate_count(flt),
                         "top": hits[0][0].doc_id if hits else "-"})
        r = {"chunker": args.chunker, "model": lever, "filters": fm,
             "retrieval": rm, "filter_mode": vm, "dims": idx.dims,
             "chunks": len(idx.chunks) or idx.doc_count(),
             "correct": correct, "of": len(QUERIES),
             "avg_candidates": round(sum(x["candidates"] for x in rows) / len(rows), 1),
             "embed_tokens": idx.embed_tokens,
             "embed_cost": round(idx.embed_tokens / 1_000_000
                                 * idx.embedder.price_per_1m_tokens, 6),
             "rows": rows}
        results.append(r)
        print("{:<14} {:<9} {:<17} {:>5}/{:<2} {:>6}".format(
            fm, rm, vm, correct, len(QUERIES), r["avg_candidates"]))
    return results


# ==========================================================================
def main() -> int:
    envlib.load_dotenv()

    ap = argparse.ArgumentParser(description="Azure AI Search index lab")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--show", metavar="QID")
    ap.add_argument("--dims", action="store_true", help="dimension truncation experiment")
    ap.add_argument("--filters", choices=list(FILTER_MODES), help="single filter mode")
    ap.add_argument("--live", choices=["embed", "search"],
                    help="embed = real Azure OpenAI vectors, lab's own index; "
                         "search = a real Azure AI Search index")
    ap.add_argument("--probe", action="store_true",
                    help="which data-plane route does the AOAI resource serve?")
    ap.add_argument("--provision", action="store_true",
                    help="--live search: create the index and upload")
    ap.add_argument("--teardown", action="store_true",
                    help="--live search: delete the index")
    ap.add_argument("--calibrate", action="store_true",
                    help="--live search: measure the per-mode score scales")
    ap.add_argument("--filter-mode",
                    choices=["preFilter", "postFilter", "strictPostFilter"],
                    help="--live search: Azure vectorFilterMode")
    ap.add_argument("--chunker", choices=list(CHUNKERS), default="semantic",
                    help="--live search: which chunker to index with")
    args = ap.parse_args()

    print("DecisionStream AI — policy index lab")
    print("=" * 76)
    print("  documents: {}   queries: {}   top_k: {}".format(
        len(DOCS), len(QUERIES), args.top_k))

    try:
        if args.probe:
            return probe_cmd()

        # ---------------- LIVE: AZURE AI SEARCH ----------------
        if args.live == "search":
            models = azure_models()
            idx, cfg, lever = search_backend(args, models)

            if args.teardown:
                gone = idx.drop()
                print("\n  index '{}' {}".format(
                    idx.name, "deleted." if gone else "did not exist."))
                return 0

            if args.provision:
                provision(idx, args.chunker)
            elif not idx.exists():
                print("\n  Index '{}' does not exist.".format(idx.name))
                print("  Create it first:  python3 run_lab.py --live search --provision")
                return 1
            else:
                idx.chunks = chunk_docs(DOCS, args.chunker)
                print("  index already exists: {} documents".format(idx.doc_count()))

            if args.calibrate:
                return calibrate(idx, args.top_k)

            if args.show:
                return show_search(idx, args)

            results = run_search_grid(idx, args)
            notes = [
                "- Backend: **Azure AI Search** `{}` index `{}`, api-version `{}`".format(
                    idx.endpoint, idx.name, idx.api_version),
                "- Embeddings: **{}** ({} dims), Azure OpenAI".format(
                    idx.embedder.deployment, idx.dims),
                "- Chunker: `{}` — {} chunks indexed".format(
                    args.chunker, idx.doc_count()),
                "- Filters are a real OData `$filter` on filterable fields; "
                "`vectorFilterMode` switches pre- vs post-filtering.",
                "- Negative-query cutoffs are per-mode because BM25, cosine and "
                "RRF are three different scales. Run `--calibrate` on your own "
                "service rather than trusting the defaults.",
            ]
            write_report(results, args.top_k, meta={
                "title": "Azure AI Search index report — DecisionStream AI (LIVE)",
                "banner": "> Live run against a real Azure AI Search index and real "
                          "Azure OpenAI embeddings.",
                "live_search": True,
                "model_desc": {lever: "**Real Azure OpenAI.** Deployment `{}`, "
                                      "{} dimensions, ${}/1M tokens.".format(
                                          idx.embedder.deployment, idx.dims,
                                          idx.embedder.price_per_1m_tokens)},
                "notes": notes})
            print_levers(results)
            print_written()
            return 0

        # ---------------- LIVE: REAL EMBEDDINGS, LAB INDEX ----------------
        models = None
        backend = "offline"
        if args.live == "embed":
            models = azure_models()
            # NOT "offline". The scoring FORMULA is still the lab's, but the
            # numbers going into it come from a different distribution, and
            # the negative-query cutoff has to move with them.
            backend = "live_embed"

        if args.calibrate:
            return calibrate_index(models, args.top_k, backend)
        if args.show:
            return show(args.show.upper(), args.top_k, models, backend)
        if args.dims:
            return dims_experiment(args.top_k, models, backend)

        fmodes = [args.filters] if args.filters else list(FILTER_MODES)
        modes = ["keyword", "vector", "hybrid"]
        results = []
        print("\n{:<11} {:<7} {:<14} {:<9} {:>8} {:>6} {:>10}".format(
            "chunker", "model", "filters", "retr", "correct", "cands", "$index"))
        print("-" * 76)
        for ch, mo, fm, rm in itertools.product(CHUNKERS, ("small", "large"),
                                                fmodes, modes):
            r = run_config(ch, mo, fm, args.top_k, mode=rm, models=models,
                           backend=backend)
            results.append(r)
            print("{:<11} {:<7} {:<14} {:<9} {:>5}/{:<2} {:>6} {:>10.6f}".format(
                ch, mo, fm, rm, r["correct"], r["of"], r["avg_candidates"],
                r["embed_cost"]))

        meta = {"model_desc": {
            "small": "**Simulated.** 384-dimension hashed *word* features. "
                     "More hash collisions, and no idea that \"bring in an "
                     "engineer\" relates to \"engineer inspection\" unless the "
                     "words literally match. Priced at $0.02/1M tokens to "
                     "mirror `text-embedding-3-small`.",
            "large": "**Simulated.** 1024-dimension hashed features over words "
                     "*plus bigrams plus a small synonym table*. Handles "
                     "paraphrase noticeably better. Priced at $0.13/1M tokens "
                     "to mirror `text-embedding-3-large`.",
        }}
        if models:
            meta = {
                "title": "Azure AI Search index report — DecisionStream AI "
                         "(LIVE embeddings)",
                "model_desc": dict(
                    (lever, "**Real Azure OpenAI.** Deployment `{}`, {} "
                            "dimensions, ${}/1M tokens.".format(
                                models[lever].deployment, models[lever].dims,
                                models[lever].price_per_1m_tokens))
                    for lever in ("small", "large")),
                "banner": "> Real Azure OpenAI embeddings, the lab's own filterable "
                          "index. Exactly one thing changed from the offline run.",
                "notes": [
                    "- Embeddings: **{}** / **{}** via Azure OpenAI ({} route)".format(
                        models["small"].deployment, models["large"].deployment,
                        models["small"].route_name()),
                    "- Total embedding spend this run: **${:.6f}** "
                    "({} tokens, {} API calls) — the disk cache in `.cache/` "
                    "means a re-run costs nothing.".format(
                        models["small"].cost_usd + models["large"].cost_usd,
                        models["small"].tokens_used + models["large"].tokens_used,
                        models["small"].api_calls + models["large"].api_calls),
                    "- The retrieval scoring is still the lab's own, so these "
                    "numbers are comparable line-for-line with the offline run.",
                ]}
        write_report(results, args.top_k, meta=meta)
        print_levers(results)
        if models:
            total = models["small"].cost_usd + models["large"].cost_usd
            calls = models["small"].api_calls + models["large"].api_calls
            print("\nEmbedding spend this run: ${:.6f} across {} API calls."
                  .format(total, calls))
            print("Cached in .cache/ — the next run costs nothing, which is why")
            print("a metadata change that forces a re-embed is the expensive kind.")
        print_written()
        return 0

    except envlib.ConfigError as e:
        print("\nCONFIG ERROR\n  {}".format(e))
        return 2
    except Exception as e:                                    # noqa: BLE001
        name = type(e).__name__
        if name in ("AzureEmbedError", "SearchError"):
            print("\nAZURE ERROR ({})\n{}".format(name, e))
            print("\nTry:  python3 run_lab.py --probe")
            return 3
        raise


def show_search(idx, args):
    q = next((x for x in QUERIES if x["id"] == args.show.upper()), None)
    if not q:
        print("No such query. Try: {}".format(", ".join(x["id"] for x in QUERIES)))
        return 1
    print("\n{}  [{}]  {}".format(q["id"], q["type"], q["q"]))
    print("  context: jurisdiction={}  decision_date={}  policy_type={}".format(
        q["jurisdiction"], q["decision_date"], q.get("policy_type")))
    print("  TRAP: {}".format(q["trap"]))
    print("  expected: {}  containing '{}'\n".format(q["expect_doc"], q["expect_text"]))
    import azure_search
    for fmode in FILTER_MODES:
        flt = FILTER_MODES[fmode](q)
        od = azure_search.odata_filter(flt)
        print("  filters={}".format(fmode))
        print("    $filter: {}".format(od or "(none)"))
        for rm in ("keyword", "vector", "hybrid"):
            for vm in (("preFilter", "postFilter") if rm != "keyword" and od
                       else ("preFilter",)):
                hits = idx.search(q["q"], args.top_k, flt, mode=rm,
                                  vector_filter_mode=vm)
                ok, why = judge(hits, q, "azure", rm)
                tops = ", ".join("{}({:.3f})".format(c.doc_id, s) for c, s in hits[:3])
                print("      {:<8} {:<11} {} {}".format(
                    rm, vm, "OK " if ok else "x  ", tops or "(nothing)"))
                if not ok:
                    print("               -> {}".format(why))
        print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
