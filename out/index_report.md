# Azure AI Search index report — DecisionStream AI

Documents: **20**   Queries: **10**   top_k: **3**

*Read the glossary first if any column heading is unfamiliar — the results are easy to misread without it.*

## How to read this report

**One row in the results table is one complete configuration of the retrieval pipeline.** Each column is an independent choice you would have to make for real, and the run varies them one at a time so you can attribute a change to a cause.

### Chunker — how a document is cut up before it is embedded

You do not embed whole documents. You cut them into chunks, embed each chunk, and retrieve chunks. Where you cut decides what a retrieved answer contains.

| Setting | What it does | Chunks from 20 docs |
|---|---|---|
| `fixed_256` | Fixed windows of **256 tokens** (~1,024 characters) with ~128 characters of overlap. Cuts wherever the window ends — including halfway through a clause. | 21 |
| `fixed_512` | Fixed windows of **512 tokens** (~2,048 characters) with ~256 characters of overlap. Same method, bigger window. | 20 |
| `semantic` | Split on the document's **own structure** — `CLAUSE 4.1`, `SECTION 2` headings. A clause stays whole and its heading travels with it, so the chunk still says what it is. | 45 |

> **Read the chunk counts before you read the chunker result.** `fixed_256` and `fixed_512` produce 21 and 20 chunks here — almost identical, because most of these policy documents are shorter than a single 1,024-character window and are never split at all. On this corpus the two fixed settings are very nearly the *same experiment*, so the small chunker swing below is partly a property of the sample, not a finding about chunk size. On real policy PDFs of 40 pages it would separate them.

### Embedding — which model turns text into a vector

| Setting | What it is |
|---|---|
| `small` | **Simulated.** 384-dimension hashed *word* features. More hash collisions, and no idea that "bring in an engineer" relates to "engineer inspection" unless the words literally match. Priced at $0.02/1M tokens to mirror `text-embedding-3-small`. |
| `large` | **Simulated.** 1024-dimension hashed features over words *plus bigrams plus a small synonym table*. Handles paraphrase noticeably better. Priced at $0.13/1M tokens to mirror `text-embedding-3-large`. |

### Filters — which metadata is applied before the search runs

**This is the axis the lab is actually testing.** The filter decides which chunks are even eligible to be compared against the query. It is applied *before* similarity, not after.

| Setting | What it does |
|---|---|
| `full` | Country **and** product **and** in-force-on-the-decision-date: `jurisdiction` + `policy_type` + `effective_from`/`effective_to` spanning the date the claim was decided. |

### Retrieval — how candidates are scored once filtered

| Setting | What it does |
|---|---|
| `keyword` | Term overlap between query and chunk, weighted by IDF so rare words count for more. Blind to meaning; excellent at clause numbers. |
| `vector` | Cosine similarity between the query embedding and each chunk embedding. Blind to identifiers like `3.5`. |
| `hybrid` | A fixed 50/50 blend of the two scores above. |

### The columns

| Column | What it means |
|---|---|
| **Correct** | Out of 10 queries. Strict: the **top-ranked** hit must be the expected document *and* contain the expected wording. A right answer at rank 2 scores zero, because a RAG pipeline that passes the top hit to a model never sees rank 2. |
| **Avg candidates** | How many chunks survived the filter, averaged over the 10 queries. **This is the column that explains the Correct column** — it is the size of the haystack the search was asked to work in. |
| **Index cost** | One-off USD to embed the whole corpus once at that chunker + embedding combination. Not a per-query cost — a per-*rebuild* cost, which is what you pay every time the metadata schema changes. |

### What 'swing' means, and why it is the number to read

**Swing** = best average minus worst average for that lever, holding the whole rest of the grid constant. It answers the only question that matters when you are deciding where to spend a sprint: *if I change this one thing, how much does the answer improve?*

A lever with a big swing is worth your time. A lever with a swing under 1 is a preference, not an engineering decision — and arguing about it in a design review is how a sprint disappears.

### What this report does NOT tell you

- **Nothing about answer quality.** It measures retrieval only — whether the right chunk reaches rank 1. What a model then writes with that chunk is a different evaluation.
- **Nothing about latency or scale.** 20 documents is a teaching corpus. HNSW, quantisation and shard counts start to matter three orders of magnitude further up.
- **Nothing about recall beyond rank 1.** A configuration scoring 3/10 here might have the right chunk at rank 2 every time. That would be a different lab, and a kinder one.

## All configurations

| Chunker | Embedding | Filters | Retrieval | Correct | Avg candidates | Index cost |
|---|---|---|---|---|---|---|
| fixed_256 | small | full | keyword | 9/10 | 1.6 | $0.000035 |
| fixed_256 | small | full | vector | 9/10 | 1.6 | $0.000035 |
| fixed_256 | small | full | hybrid | 9/10 | 1.6 | $0.000035 |
| fixed_256 | large | full | keyword | 9/10 | 1.6 | $0.000230 |
| fixed_256 | large | full | vector | 9/10 | 1.6 | $0.000230 |
| fixed_256 | large | full | hybrid | 9/10 | 1.6 | $0.000230 |
| fixed_512 | small | full | keyword | 9/10 | 1.0 | $0.000035 |
| fixed_512 | small | full | vector | 9/10 | 1.0 | $0.000035 |
| fixed_512 | small | full | hybrid | 9/10 | 1.0 | $0.000035 |
| fixed_512 | large | full | keyword | 9/10 | 1.0 | $0.000227 |
| fixed_512 | large | full | vector | 9/10 | 1.0 | $0.000227 |
| fixed_512 | large | full | hybrid | 9/10 | 1.0 | $0.000227 |
| semantic | small | full | keyword | 9/10 | 4.2 | $0.000035 |
| semantic | small | full | vector | 9/10 | 4.2 | $0.000035 |
| semantic | small | full | hybrid | 9/10 | 4.2 | $0.000035 |
| semantic | large | full | keyword | 9/10 | 4.2 | $0.000226 |
| semantic | large | full | vector | 9/10 | 4.2 | $0.000226 |
| semantic | large | full | hybrid | 9/10 | 4.2 | $0.000226 |

## Which lever actually moved the result

Average correctness, holding everything else across the grid:

| Lever | Setting | Avg correct |
|---|---|---|
| retrieval | keyword | 9.0 |
| retrieval | vector | 9.0 |
| retrieval | hybrid | 9.0 |
| model | small | 9.0 |
| model | large | 9.0 |
| chunker | fixed_256 | 9.0 |
| chunker | fixed_512 | 9.0 |
| chunker | semantic | 9.0 |

### Read the swing, not the winner

- Changing the **retrieval** moves correctness by **0.00** queries
- Changing the **model** moves correctness by **0.00** queries
- Changing the **chunker** moves correctness by **0.00** queries

Metadata filtering is not a tuning parameter. It is the difference between answering about the right country and the right year, and answering fluently about the wrong one. No embedding model closes that gap, because it is not a similarity problem.

## Best configuration

**fixed_256 + small + filters=full** — 9/10 correct, 21 chunks, index cost $0.000035.

## Questions to answer before this goes in the ADR

1. Which metadata fields are REQUIRED at ingestion, and what happens to a document that arrives without them?
2. Q06 asks about a claim decided in June 2025 and the current policy is the wrong answer. How does your index know the decision date?
3. Your corpus contains superseded versions. Should they be indexed at all — and what breaks if you delete them?
4. What does the embedding cost look like on a full re-index, and how often will you re-index?
5. Which of your ten queries would still fail, and who finds out?
6. What is your review trigger for rebuilding this index?
