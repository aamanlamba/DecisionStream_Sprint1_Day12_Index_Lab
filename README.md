# Sprint 1 · Day 12 Lab — Policy Index and Metadata Filtering

**Birlasoft FORGE FDE Academy · Confidential — For Learner Use Only**

This lab shows how to take 20 business policy documents, split them into smaller
chunks, create embeddings, and build a searchable index with metadata filters.

It also has 10 questions to check whether adding metadata filters actually helps.

---

## Run the lab offline

You can run the lab without internet, Azure, API keys, or installing any packages:

```bash
python3 run_lab.py
```

Only Python's standard library is used. There is no Azure SDK and no `pip install`.

The same lab can also run against real Azure in two steps. It uses the same
documents, the same 10 questions, and the same scoring method, so you can compare
the results directly. See [Going live](#going-live).

### No `requirements.txt` or virtual environment is needed

This is intentional. It is also true for live mode.

The Azure calls use normal REST requests through `urllib`. They do not use the
OpenAI or `azure-search-documents` SDKs, so there is nothing extra to install.

| | |
|---|---|
| Third-party imports | **None** — only Python standard library modules are used |
| Minimum Python | **3.7** (tested on 3.7.2 and 3.9.6; learner VMs use 3.12) |
| Tested with | A virtual environment made with `--without-pip`, with zero packages, both offline and with real Azure |

The disadvantage is that this lab has its own request and retry code instead of
using an SDK.

The advantage is that it can run on a locked-down VM or behind a proxy without
an installation step. You can also see every HTTP call, which is useful for
understanding how the index works.

> Note: The Day-11 lab uses `requirements.txt` because it uses the OpenAI SDK.
> The project-wide `.venv` rule is for the Excel authoring scripts and does not
> apply to this lab.

---

## The most important result

Run the full test and look at the last four lines:

```text
AVERAGE CORRECTNESS BY LEVER
  filters    full=9.00   jurisdiction=4.72   none=2.72     swing 6.28
  retrieval  keyword=5.67  hybrid=5.67  vector=5.11        swing 0.56
  model      large=5.52  small=5.44                        swing 0.07
  chunker    semantic=5.56  fixed_256=5.44  fixed_512=5.44 swing 0.11
```

### Main learning

**Metadata filtering improves the result by about 6 questions.
Everything else improves it by less than 1 question.**

This happens because a policy library can contain old versions and policies from
other countries or regions. These documents may be well written and easy to find,
but they can still be the wrong answer for the question.

> An embedding model cannot solve this problem because this is not only a
> similarity problem. Filtering makes sure we search the correct country and
> correct year, instead of giving a good-sounding answer about the wrong one.

---

## Why are wrong answers included?

The lab intentionally keeps old and different policies in the index.

For the same subject — the motor total loss threshold — the index contains:

| Document | Answer |
|---|---:|
| Motor, UK, 2024 | 70% |
| Motor, UK, 2025 | 65% |
| **Motor, UK, 2026** | **60%** ← current policy |
| Motor, Ireland, 2026 | 55% |
| Motor, EU, 2026 | 65% |
| Property, UK, 2026 | 80% |

All six answers may look correct. But for a UK motor claim being decided today,
only one is correct.

We do not delete the 2024 policy because a claim decided in 2024 may still need
to be explained using the rules that were valid at that time.

---

## What you need to do

### 1. Try without metadata filters — 5 minutes

```bash
python3 run_lab.py --filters none
```

You should get about 3 correct answers out of 10.

Open `out/per_query.csv` and check which document was selected.

### 2. Add only the country/jurisdiction filter — 5 minutes

```bash
python3 run_lab.py --filters jurisdiction
```

You should get about 5 correct answers out of 10.

This is better, but it can still select an old policy from the correct country.

### 3. Add effective date and product — 5 minutes

```bash
python3 run_lab.py --filters full
```

You should get about 9 correct answers out of 10.

**This is the main lesson of the lab.**

### 4. Study one question in detail — 10 minutes

```bash
python3 run_lab.py --show Q01     # version problem
python3 run_lab.py --show Q06     # historical question
python3 run_lab.py --show Q08     # paraphrase question
```

**Q06 is especially important.**

It asks about a claim decided in June 2025. The current policy is not the correct
answer.

A decision should use the rules that were valid on the date when the decision was
made. That is why `effective_from` and `effective_to` are important index fields,
not just documentation fields.

### 5. Test different embedding dimensions — 10 minutes

```bash
python3 run_lab.py --dims
```

The result shows that reducing dimensions has little effect when your filters are
good, but it can hurt more when the filters are not good.

In this lab, performance stays reasonable down to about 128 dimensions and then
drops quickly.

### 6. Answer the six questions

Find the answers in:

```text
out/index_report.md
```

---

## The 10 questions

| ID | Type | What it tests |
|---|---|---|
| Q01 | Version problem | Four old or foreign answers compete |
| Q02 | Version problem | Young driver excess changed twice in three years |
| Q03 | Country problem | Ireland says 28 days; UK and EU say 30 |
| Q04 | Country problem | Same question as Q01, but for another country |
| Q05 | Type problem | Property has the same clause number but a different rule |
| Q06 | **Historical** | The current policy is the wrong answer |
| Q07 | Paraphrase | The question and policy use very different words |
| Q08 | Paraphrase | The question and policy have no words in common |
| Q09 | Clause lookup | A clause number identifies something, but does not tell its meaning |
| Q10 | Negative | The answer is not in the documents. Does the system correctly give a weak result? |

**Q08 fails in every setup.**

This is intentional. The paraphrase is so different from the original policy
wording that the simulated embedding cannot connect them.

If you run `--show Q08`, you will see that the large vector model ranks the correct
chunk first, but hybrid search then combines it with keyword results.

So, **hybrid search is not always better**. This is useful to understand before
using it in Azure.

---

## Files in this lab

```text
run_lab.py                 The main command you run
index_lab/
  corpus.py                20 policy documents, metadata, and 10 questions
  indexing.py              Chunking, simulated embeddings, and searchable index
  env.py                   Reads .env and prepares endpoints
  azure_embed.py           Real Azure OpenAI embeddings and cache
  azure_search.py          Real Azure AI Search index, filters, and vectors
.env.example               Example settings for live mode
out/                       Reports and test results
```

---

<a name="going-live"></a>

## Going live with Azure

Offline mode simulates two parts:

1. Embeddings
2. The search index

You can replace these one at a time. This makes it easier to understand which
change caused a different result.

First create your `.env` file:

```bash
cp .env.example .env
```

Then fill in the required values.

### Layer 1 — Use real embeddings

You can first use real Azure embeddings while keeping the lab's own index:

```bash
python3 run_lab.py --probe
python3 run_lab.py --live embed
```

The first command checks which Azure route your resource supports.

The second command uses real `text-embedding-3-*` vectors. Everything else stays
the same: chunking, filtering, scoring, and judging.

The expectation is:

- The embedding model can improve the **model** result.
- It should not change the **filter** result.

If the filter result changes a lot, check your setup.

Embeddings are saved in `.cache/`, so the second run does not need to create them
again.

This also shows why changing your metadata design later can be expensive:
re-embedding can take time and money.

### Layer 2 — Use a real Azure AI Search index

Create and test a real Azure Search index:

```bash
python3 run_lab.py --live search --provision
python3 run_lab.py --live search
python3 run_lab.py --live search --calibrate
python3 run_lab.py --live search --show Q06
python3 run_lab.py --live search --teardown
```

These commands do the following:

- `--provision` creates the index, creates embeddings, and uploads the documents.
- No extra option runs the test.
- `--calibrate` checks the score ranges.
- `--show Q06` shows one question and its filter.
- `--teardown` deletes the test index.

In live mode:

- Filtering uses a real OData `$filter`.
- Keyword search uses BM25.
- Vector search uses HNSW with cosine similarity.
- Hybrid search uses RRF to combine results.

---

## Important: `vectorFilterMode`

One of the most important settings in Azure AI Search is `vectorFilterMode`.

It has three options:

| Setting | What it does | Risk |
|---|---|---|
| `preFilter` (default) | Applies the filter while searching each shard | None — returns k results if matching documents exist |
| `postFilter` | Searches each shard first, then applies the filter | Moderate — some matching documents may be missed |
| `strictPostFilter` | Finds the overall top-k results first, then applies the filter | Highest — can return zero results even when matching documents exist |

In this lab, for `jurisdiction eq 'IE'` with 5 out of 45 chunks matching and
`k=3`:

```text
preFilter          3 results
postFilter         3 results
strictPostFilter   0 results
```

### Why does `postFilter` look the same here?

This lab has only 45 chunks, so there are not enough documents or shards to show
a big difference.

Microsoft's benchmark shows that the difference becomes more visible on much
larger indexes.

Also, Azure Search tiers have different partition limits:

- Basic: 1 partition by default, up to 3
- S1/S2/S3: up to 12 partitions

So this lab cannot properly measure the large-scale `postFilter` behaviour on
Basic. It explains the behaviour instead of pretending that a small test proves it.

### Why is `strictPostFilter` important?

It first finds the global top-k results without filtering.

If those top results belong to the wrong country, they are removed by the filter.
The correct document may have ranked lower and therefore never gets a chance.

In this lab, it can lose up to **4 out of 10 questions**.

Try it:

```bash
python3 run_lab.py --live search --filter-mode strictPostFilter
```

The lab compares these modes fairly against `preFilter`.

---

## Three important things you learn from the live run

### 1. `filterable` is a design decision

You must decide when creating the index which fields can be used for filtering.

You cannot filter on a field that was not marked as `filterable`.

If you need to add it later, you may have to delete and recreate the index and
re-embed the chunks. This can cost real money on a large system.

The fields used in this lab are:

```text
policy_type      filterable, facetable
jurisdiction     filterable, facetable
effective_from   filterable, sortable
effective_to     filterable, sortable, nullable
superseded       filterable, facetable
version          retrievable
contentVector    searchable
```

### 2. Be careful with `null`

The current policy has:

```text
effective_to = null
```

A simple comparison can accidentally remove the current policy.

Wrong:

```sql
effective_to ge 2026-04-01T00:00:00Z
```

Right:

```sql
(effective_to eq null or effective_to ge 2026-04-01T00:00:00Z)
```

This kind of mistake does not give an error. It can simply return a confident,
but wrong, answer.

### 3. Keyword, vector, and hybrid scores are different

Run:

```bash
python3 run_lab.py --live search --calibrate
```

The scores measured in this lab are:

| Mode | Question the corpus can answer | Question it cannot answer | Difference |
|---|---:|---:|---:|
| keyword (BM25) | 3.496 | 0.000 | **3.496** |
| vector (cosine) | 0.655 | 0.593 | **0.062** |
| hybrid (RRF) | 0.0333 | 0.0167 | **0.0167** |

This means you should **not use one common score limit for all three search
types**.

For example, a relevance rule such as "score above 0.7 means relevant" may work
for one search type but be completely wrong for another.

That is why the negative-query cutoff in this lab is configured separately for
each search mode.

Another important point: vector search has a very small difference between a
question the corpus can answer and one it cannot.

So do not depend only on a cosine score to decide:

> "We don't have an answer for this."

Keyword search gives a much clearer separation here because a document with no
matching terms gets a score of zero.

---

## Other important issues already handled by the lab

- **Document IDs:** Azure allows only certain characters in document keys.
  The lab cleans the chunk IDs and keeps the original value in `chunk_id`.
- **Indexing is asynchronous:** A document may be accepted before it can be
  searched. The lab waits until the documents are available.
- **Azure OpenAI has two API routes:** Using the wrong route can cause
  `DeploymentNotFound`, even when the deployment exists. `--probe` checks which
  route your resource supports.
- **Index limits depend on the Azure tier:** Free supports 3 indexes, Basic 15,
  S1 50, and S2/S3 200. The live test creates one index deliberately.

---

## What does your Azure tier give you?

The Azure CLI may not be able to list resources in a locked-down learner
subscription.

The data plane can still report the service information using:

```text
GET /servicestats
```

This needs the admin key.

Some example quotas are:

| Quota | Tier |
|---|---|
| 15 indexes · 15 GB storage · **5 GB vector** | Basic |
| 50 indexes · 160 GB storage · **35 GB vector** | S1 |

The lab prints this information during a live search run.

### Vector storage

Vector storage can become the first limit you hit.

In this lab, 45 chunks using `text-embedding-3-large` at 3,072 dimensions use
about 556 KB of vector index storage.

Approximate capacity for a Basic service with 5 GB vector storage:

| Dimensions | Chunks that fit | Approx. policy documents |
|---|---:|---:|
| 3,072 (`3-large` native) | ~434,000 | ~193,000 |
| 1,536 (`3-small`, or `3-large` truncated) | ~868,000 | ~386,000 |
| 1,024 | ~1,300,000 | ~579,000 |
| 512 | ~2,600,000 | ~1,157,000 |

Reducing the dimensions roughly doubles the number of chunks that can fit.

The `--dims` test helps you understand what you lose in correctness when you
reduce dimensions.

For this lab, reducing dimensions has little impact when the filters are good.
This supports the main point: **good metadata can be more useful than simply
buying a bigger Azure tier.**

This lab uses only about **0.01%** of the Basic vector quota. The quota becomes
important at roughly 400,000 chunks.

> Basic does not change the search ranking in this lab. BM25, HNSW, cosine,
> RRF, OData filters, and `preFilter`/`strictPostFilter` work the same way.
> Higher tiers mainly give you more capacity, throughput, and partitions.

### Check Azure versions before building

This lab was tested with Search REST API `2026-04-01` on **12 Aug 2026**.

Azure APIs can change. For example, `vectorQueries` was a breaking change from
the older `2023-07-01` preview format.

**Always check the current Azure service, SDK, and API documentation before
building a production system.**

---

## One important limitation of offline mode

The offline embeddings are **simulated**.

They are made using hashed feature vectors:

- The small model uses words.
- The large model uses words, bigrams, and a small synonym list.

This copies the general behaviour we want to demonstrate, but it is not the same
quality as a real embedding model.

A real `text-embedding-3-large` model should perform better on Q07 and Q08.

You can test the difference:

```bash
python3 run_lab.py                 # simulated embeddings
python3 run_lab.py --live embed    # real embeddings
```

The important point does not change:

**A better embedding model may improve similarity search, but it cannot know the
correct country or the correct date for a policy. That information must come from
metadata and filters.**

---

## Before you present this lab

Use this checklist:

- [ ] You can explain the result of each of the four levers.
- [ ] You can explain Q06 in one sentence.
- [ ] You can say which metadata fields are needed when documents are added.
- [ ] You can explain what happens when a document does not have the required metadata.
- [ ] You know what re-indexing costs and when you may need to do it.
- [ ] You can name the question that still fails and explain why.
- [ ] You can explain `preFilter` vs `postFilter` and say which one you prefer.
- [ ] You can explain why one relevance score limit cannot be used for keyword,
      vector, and hybrid search.
