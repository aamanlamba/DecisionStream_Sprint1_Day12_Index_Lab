# Azure AI Search index report — DecisionStream AI (LIVE)

> Live run against a real Azure AI Search index and real Azure OpenAI embeddings.

Documents: **20**   Queries: **10**   top_k: **3**

*Read the glossary first if any column heading is unfamiliar — the results are easy to misread without it.*

## How to read this report

**One row in the results table is one complete configuration of the retrieval pipeline.** Each column is an independent choice you would have to make for real, and the run varies them one at a time so you can attribute a change to a cause.

### Chunker — how a document is cut up before it is embedded

You do not embed whole documents. You cut them into chunks, embed each chunk, and retrieve chunks. Where you cut decides what a retrieved answer contains.

| Setting | What it does | Chunks from 20 docs |
|---|---|---|
| `semantic` | Split on the document's **own structure** — `CLAUSE 4.1`, `SECTION 2` headings. A clause stays whole and its heading travels with it, so the chunk still says what it is. | 45 |

### Embedding — which model turns text into a vector

| Setting | What it is |
|---|---|
| `large` | **Real Azure OpenAI.** Deployment `text-embedding-3-large`, 3072 dimensions, $0.13/1M tokens. |

### Filters — which metadata is applied before the search runs

**This is the axis the lab is actually testing.** The filter decides which chunks are even eligible to be compared against the query. It is applied *before* similarity, not after.

| Setting | What it does |
|---|---|
| `none` | No metadata at all. Every chunk in the corpus is a candidate. This is what you get if you index the text and nothing else. |
| `jurisdiction` | Filter on country only — `jurisdiction eq 'UK'`. The obvious half-measure. |
| `full` | Country **and** product **and** in-force-on-the-decision-date: `jurisdiction` + `policy_type` + `effective_from`/`effective_to` spanning the date the claim was decided. |

### Retrieval — how candidates are scored once filtered

| Setting | What it does |
|---|---|
| `keyword` | Real **BM25** full-text search over the `content` field. |
| `vector` | Real **HNSW** approximate-nearest-neighbour search over cosine distance — run exhaustively here, because on 45 chunks the approximation has nothing to approximate and exhaustive keeps the lab reproducible. In production you want the approximation; it is what you are paying HNSW for. |
| `hybrid` | Both, fused by **Reciprocal Rank Fusion (RRF)** — Azure combines the two ranked lists by position, not by score. |

### vectorFilterMode — *when* the filter is applied

| Setting | What it does | Risk |
|---|---|---|
| `preFilter` | Applies the filter **during** HNSW traversal on each shard, expanding the graph until k candidates are found. Default since ~Oct 2023, and recommended. | None — guarantees k results if they exist |
| `postFilter` | Traverses each shard **without** the filter to get that shard's local top-k, filters *that*, then aggregates. | Moderate — misses matches per shard |
| `strictPostFilter` | Finds the **unfiltered global top-k** first, then filters it. Preview; needs a preview api-version. | Highest — can return **zero** results when matches exist |

> **`postFilter` is a no-op on a corpus this small, and that is not a bug.** It degrades *per shard*, and 45 chunks is one shard whose local top-k is effectively the whole set. Microsoft's own benchmark calls the two modes "approximately equal" on small indexes; the divergence needs 10^5–10^6 documents across several shards. `strictPostFilter` filters the *global* top-k, so it fails at any size — which is why it is the one that demonstrates the point here. Do not generalise its result into a claim about `postFilter`.

### The columns

| Column | What it means |
|---|---|
| **Correct** | Out of 10 queries. Strict: the **top-ranked** hit must be the expected document *and* contain the expected wording. A right answer at rank 2 scores zero, because a RAG pipeline that passes the top hit to a model never sees rank 2. |
| **Avg candidates** | How many chunks survived the filter, averaged over the 10 queries. **This is the column that explains the Correct column** — it is the size of the haystack the search was asked to work in. |
| **Index cost** | One-off USD to embed the whole corpus once at that chunker + embedding combination. Not a per-query cost — a per-*rebuild* cost, which is what you pay every time the metadata schema changes. |

### Worked example — reading one pair of rows

```
chunker=semantic  embedding=large  retrieval=keyword  filters=none  -> 4/10 correct, 45.0 candidates
chunker=semantic  embedding=large  retrieval=keyword  filters=full  -> 9/10 correct, 4.2 candidates
```

Same chunker. Same embedding model. Same retrieval method. The **only** difference is the metadata filter, and it is worth **5 queries out of 10**.

The candidate count is why. It fell from about 45 chunks to about 4: the search stopped having to tell the right clause apart from roughly 41 other well-written candidates, because those were never eligible to be compared in the first place. Nothing about the search got smarter — the haystack got smaller.

### What 'swing' means, and why it is the number to read

**Swing** = best average minus worst average for that lever, holding the whole rest of the grid constant. It answers the only question that matters when you are deciding where to spend a sprint: *if I change this one thing, how much does the answer improve?*

A lever with a big swing is worth your time. A lever with a swing under 1 is a preference, not an engineering decision — and arguing about it in a design review is how a sprint disappears.

### What this report does NOT tell you

- **Nothing about answer quality.** It measures retrieval only — whether the right chunk reaches rank 1. What a model then writes with that chunk is a different evaluation.
- **Nothing about latency or scale.** 20 documents is a teaching corpus. HNSW, quantisation and shard counts start to matter three orders of magnitude further up.
- **Nothing about recall beyond rank 1.** A configuration scoring 3/10 here might have the right chunk at rank 2 every time. That would be a different lab, and a kinder one.

## All configurations

| Chunker | Embedding | Filters | vectorFilterMode | Retrieval | Correct | Avg candidates | Index cost |
|---|---|---|---|---|---|---|---|
| semantic | large | full | preFilter | keyword | 9/10 | 4.2 | $0.000000 |
| semantic | large | full | preFilter | hybrid | 9/10 | 4.2 | $0.000000 |
| semantic | large | full | postFilter | hybrid | 9/10 | 4.2 | $0.000000 |
| semantic | large | full | preFilter | vector | 8/10 | 4.2 | $0.000000 |
| semantic | large | full | postFilter | vector | 8/10 | 4.2 | $0.000000 |
| semantic | large | full | strictPostFilter | hybrid | 8/10 | 4.2 | $0.000000 |
| semantic | large | jurisdiction | preFilter | keyword | 6/10 | 29.0 | $0.000000 |
| semantic | large | none | preFilter | keyword | 4/10 | 45.0 | $0.000000 |
| semantic | large | jurisdiction | preFilter | hybrid | 4/10 | 29.0 | $0.000000 |
| semantic | large | jurisdiction | postFilter | hybrid | 4/10 | 29.0 | $0.000000 |
| semantic | large | jurisdiction | strictPostFilter | hybrid | 4/10 | 29.0 | $0.000000 |
| semantic | large | full | strictPostFilter | vector | 4/10 | 4.2 | $0.000000 |
| semantic | large | jurisdiction | preFilter | vector | 3/10 | 29.0 | $0.000000 |
| semantic | large | jurisdiction | postFilter | vector | 3/10 | 29.0 | $0.000000 |
| semantic | large | none | preFilter | hybrid | 2/10 | 45.0 | $0.000000 |
| semantic | large | jurisdiction | strictPostFilter | vector | 2/10 | 29.0 | $0.000000 |
| semantic | large | none | preFilter | vector | 1/10 | 45.0 | $0.000000 |

## Which lever actually moved the result

Average correctness, holding everything else across the grid:

| Lever | Setting | Avg correct |
|---|---|---|
| filters | full | 8.67 |
| filters | jurisdiction | 4.33 |
| filters | none | 2.33 |
| retrieval | keyword | 6.33 |
| retrieval | hybrid | 5.0 |
| retrieval | vector | 4.0 |

### Read the swing, not the winner

- Changing the **filters** moves correctness by **6.34** queries
- Changing the **retrieval** moves correctness by **2.33** queries

Metadata filtering is not a tuning parameter. It is the difference between answering about the right country and the right year, and answering fluently about the wrong one. No embedding model closes that gap, because it is not a similarity problem.

## Best configuration

**semantic + large + filters=full** — 9/10 correct, 45 chunks, index cost $0.000000.

## vectorFilterMode, paired against preFilter

Only configurations where both modes ran. The post-filter modes are skipped for keyword search and for `filters=none`, where they mean nothing — so an unpaired average would compare different subsets and flatter them by omitting the rows they break.

| Filters | Retrieval | Mode | preFilter | That mode | Delta |
|---|---|---|---|---|---|
| full | vector | `strictPostFilter` | 8/10 | 4/10 | -4 |
| full | hybrid | `strictPostFilter` | 9/10 | 8/10 | -1 |
| jurisdiction | vector | `strictPostFilter` | 3/10 | 2/10 | -1 |
| full | hybrid | `postFilter` | 9/10 | 9/10 | +0 |
| full | vector | `postFilter` | 8/10 | 8/10 | +0 |
| jurisdiction | hybrid | `postFilter` | 4/10 | 4/10 | +0 |
| jurisdiction | vector | `postFilter` | 3/10 | 3/10 | +0 |
| jurisdiction | hybrid | `strictPostFilter` | 4/10 | 4/10 | +0 |

**`postFilter` shows no difference here, and that is the honest result rather than a demonstration.** It degrades *per shard*; 45 chunks is a single shard whose local top-k is effectively the whole set. This lab cannot measure that mode — it can only explain it.

**`strictPostFilter` does fail here, by up to 4 queries.** It filters the *global* top-k, so it breaks at any corpus size: the k nearest across the whole index are the wrong jurisdiction, they are discarded, and the right clause — which ranked below them — was never in the running. You ask for 3 results and get 0.

The filter is not a post-processing step. It decides what is eligible to be compared at all.

## What this run actually used

- Backend: **Azure AI Search** `https://bsoftaisearchaa10.search.windows.net` index `decisionstream-policy`, api-version `2026-04-01`
- Embeddings: **text-embedding-3-large** (3072 dims), Azure OpenAI
- Chunker: `semantic` — 45 chunks indexed
- Filters are a real OData `$filter` on filterable fields; `vectorFilterMode` switches pre- vs post-filtering.
- Negative-query cutoffs are per-mode because BM25, cosine and RRF are three different scales. Run `--calibrate` on your own service rather than trusting the defaults.

## Questions to answer before this goes in the ADR

1. Which metadata fields are REQUIRED at ingestion, and what happens to a document that arrives without them?
2. Q06 asks about a claim decided in June 2025 and the current policy is the wrong answer. How does your index know the decision date?
3. Your corpus contains superseded versions. Should they be indexed at all — and what breaks if you delete them?
4. What does the embedding cost look like on a full re-index, and how often will you re-index?
5. Which of your ten queries would still fail, and who finds out?
6. What is your review trigger for rebuilding this index?
