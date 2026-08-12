"""
indexing.py — chunking, embeddings, and a vector index that supports filters.

Standard library only. Every mechanic here is what Azure AI Search does for
you; building it once means you know what the configuration settings mean.

THREE THINGS THIS FILE DEMONSTRATES
-----------------------------------
1. CHUNKING       fixed-size versus semantic (clause boundaries)
2. EMBEDDINGS     a small model and a large one, plus dimension truncation
3. FILTERING      metadata applied BEFORE search, not after

The third one is today's lesson, and the lab measures it.

A NOTE ON THE SIMULATED EMBEDDINGS
----------------------------------
Real embeddings come from a trained model. These are hashed feature vectors:

  small_model  — word features only, 384 dimensions.
                 More hash collisions, and no idea that "bring in an engineer"
                 and "engineer inspection" are related unless the words match.

  large_model  — words + bigrams + a small synonym expansion, 1024 dimensions.
                 Handles paraphrase noticeably better. Costs more per token.

That is the SHAPE of the real difference, not a substitute for it. A real
text-embedding-3-large genuinely does better on paraphrase and genuinely
costs more. What it will not do — and no embedding model will — is know
which jurisdiction your claim is in.

DIMENSION TRUNCATION
--------------------
text-embedding-3 models are trained so that a truncated vector still works,
degrading gradually rather than falling over. That is a real cost lever:
fewer dimensions means a smaller index and faster search. The lab lets you
truncate and measure what it costs you in recall, which is the only way to
decide whether the saving is worth it.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from corpus import PolicyDoc


STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
        "is", "are", "was", "were", "be", "by", "with", "as", "that", "this",
        "it", "from", "any", "not", "no", "which", "where", "under", "does",
        "do", "what", "how", "we", "i", "my", "our", "can", "must", "may"}

# A small expansion table. A real embedding model learns relationships like
# these from data; here they are written down so the effect is visible and
# explainable rather than magic.
SYNONYMS = {
    "engineer": ["inspection", "assess", "surveyor"],
    "extra": ["supplementary", "additional"],
    "later": ["supplementary", "after"],
    "decide": ["decision", "automated", "authorise"],
    "person": ["human", "handler"],
    "notify": ["notification", "notified"],
    "days": ["days", "working"],
    "threshold": ["exceeds", "above", "limit"],
    "car": ["vehicle", "motor"],
    "age": ["years", "under", "driver"],
    "money": ["gbp", "eur", "excess"],
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\.\-]*", text.lower())
            if t not in STOP]


def approx_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


# ==========================================================================
# CHUNKING
# ==========================================================================
@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    n_tokens: int
    meta: dict


def chunk_fixed(doc: PolicyDoc, target_tokens: int) -> list[Chunk]:
    """Fixed-size with overlap. Fast, predictable, and it will cut a clause
    in half without noticing."""
    chars = target_tokens * 4
    overlap = chars // 8
    text = doc.text.strip()
    out, i, n = [], 0, 0
    while i < len(text):
        piece = text[i:i + chars].strip()
        if piece:
            out.append(Chunk(f"{doc.id}#f{target_tokens}-{n}", doc.id, piece,
                             approx_tokens(piece), doc.meta()))
        i += max(1, chars - overlap)
        n += 1
    return out


HEAD_RE = re.compile(r"^(CLAUSE\s+[\d\.]+|SECTION\s+\d+)[^\n]*$", re.MULTILINE)


def chunk_semantic(doc: PolicyDoc, _t: int = 0) -> list[Chunk]:
    """Split on the document's own boundaries — clause and section headings.
    A clause stays whole, and the heading travels with its body so the chunk
    still says what it is."""
    text = doc.text.strip()
    marks = [m.start() for m in HEAD_RE.finditer(text)]
    if len(marks) >= 2:
        bounds = marks + [len(text)]
        pieces = [text[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    else:
        pieces = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [Chunk(f"{doc.id}#s-{i}", doc.id, p.strip(), approx_tokens(p), doc.meta())
            for i, p in enumerate(pieces) if p.strip()]


CHUNKERS = {
    "fixed_256": lambda d: chunk_fixed(d, 256),
    "fixed_512": lambda d: chunk_fixed(d, 512),
    "semantic":  chunk_semantic,
}


# ==========================================================================
# EMBEDDINGS
# ==========================================================================
@dataclass
class EmbeddingModel:
    name: str
    dims: int
    use_bigrams: bool
    use_synonyms: bool
    price_per_1m_tokens: float      # illustrative — look up current pricing

    def features(self, text: str) -> list[str]:
        toks = tokenize(text)
        feats = list(toks)
        if self.use_bigrams:
            feats += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        if self.use_synonyms:
            for t in toks:
                feats += SYNONYMS.get(t, [])
        return feats

    def embed(self, text: str, dims: int | None = None) -> list[float]:
        d = dims or self.dims
        vec = [0.0] * d
        for f in self.features(text):
            h = int(hashlib.sha256(f.encode()).hexdigest()[:16], 16)
            vec[h % d] += 1.0 if (h >> 63) & 1 == 0 else -1.0
        n = math.sqrt(sum(v * v for v in vec)) or 1e-9
        return [v / n for v in vec]


MODELS = {
    "small": EmbeddingModel("small", 384, False, False, 0.02),
    "large": EmbeddingModel("large", 1024, True, True, 0.13),
}


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ==========================================================================
# THE INDEX
# ==========================================================================
class VectorIndex:
    """
    A vector index with metadata filtering.

    The important method is `search`, and the important argument is `flt`.
    Filtering happens BEFORE similarity is computed — the same way Azure AI
    Search applies a filter expression. Filtering afterwards would mean
    top-k is filled with documents you then throw away, and the answer you
    wanted never makes the shortlist.
    """

    def __init__(self, chunks: list[Chunk], model: EmbeddingModel,
                 dims: int | None = None, progress=None):
        self.chunks = chunks
        self.model = model
        self.dims = dims or model.dims
        # A real embedding model batches. Calling it once per chunk turns a
        # two-round-trip index build into three hundred, and the difference
        # is the whole of your build time. The simulated model has no
        # embed_all because there is nothing to batch.
        if hasattr(model, "embed_all"):
            self.vecs = model.embed_all([c.text for c in chunks],
                                        dims=self.dims, progress=progress)
        else:
            self.vecs = [model.embed(c.text, self.dims) for c in chunks]
        # Keyword side, so we can show hybrid and why clause numbers need it.
        self.toks = [set(tokenize(c.text)) for c in chunks]
        df: dict[str, int] = {}
        for t in self.toks:
            for w in t:
                df[w] = df.get(w, 0) + 1
        n = max(1, len(chunks))
        self.idf = {w: math.log(1 + n / (1 + c)) for w, c in df.items()}
        self.embed_tokens = sum(c.n_tokens for c in chunks)

    def _passes(self, c: Chunk, flt: dict | None) -> bool:
        if not flt:
            return True
        m = c.meta
        if flt.get("jurisdiction") and m["jurisdiction"] != flt["jurisdiction"]:
            return False
        if flt.get("policy_type") and m["policy_type"] != flt["policy_type"]:
            return False
        on = flt.get("in_force_on")
        if on:
            if m["effective_from"] > on:
                return False
            if m["effective_to"] and m["effective_to"] < on:
                return False
        return True

    def search(self, query: str, top_k: int, flt: dict | None = None,
               mode: str = "vector") -> list[tuple[Chunk, float]]:
        qv = self.model.embed(query, self.dims)
        qt = set(tokenize(query))
        out = []
        for i, c in enumerate(self.chunks):
            if not self._passes(c, flt):
                continue
            if mode == "vector":
                sc = cosine(qv, self.vecs[i])
            elif mode == "keyword":
                sc = sum(self.idf.get(w, 0.0) for w in qt & self.toks[i])
            else:  # hybrid
                v = cosine(qv, self.vecs[i])
                k = sum(self.idf.get(w, 0.0) for w in qt & self.toks[i])
                sc = 0.5 * v + 0.5 * (k / 10.0)
            out.append((c, sc))
        out.sort(key=lambda x: -x[1])
        return out[:top_k]

    def candidate_count(self, flt: dict | None) -> int:
        return sum(1 for c in self.chunks if self._passes(c, flt))


def chunk_docs(docs: list[PolicyDoc], chunker: str) -> list[Chunk]:
    fn = CHUNKERS[chunker]
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(fn(d))
    return chunks


def build_index(docs: list[PolicyDoc], chunker: str, model_name: str,
                dims: int | None = None, models: dict | None = None,
                progress=None) -> VectorIndex:
    """`models` swaps the simulated embedding pair for the real Azure one.
    Nothing else in the lab changes — which is the point of the experiment."""
    registry = models if models is not None else MODELS
    return VectorIndex(chunk_docs(docs, chunker), registry[model_name],
                       dims, progress=progress)
