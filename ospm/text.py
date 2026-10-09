"""Small, dependency-free text utilities: tokenization, phrase matching and TF-IDF retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just let me more
    most my myself no nor not now of off on once only or other our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there these they this those
    through to too under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves also get got use using used would like one two im ive dont
    cant doesnt isnt wasnt thanks thank please hi hello hey
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric/underscore tokens. Punctuation is treated as a separator."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stop-words and very short/numeric-only tokens removed."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1 and not t.isdigit()]


def normalize(text: str) -> str:
    """Space-delimited token string with sentinel spaces, used for phrase matching."""
    return " " + " ".join(tokenize(text)) + " "


def phrase_in(phrase: str, normalized_text: str) -> bool:
    """True if the token sequence of `phrase` occurs contiguously in `normalized_text`.

    Both sides are tokenized identically, so "shell=True" matches the phrase "shell true".
    """
    toks = tokenize(phrase)
    if not toks:
        return False
    return (" " + " ".join(toks) + " ") in normalized_text


def ngrams(tokens: list[str], n_max: int = 2) -> list[str]:
    out: list[str] = []
    for n in range(1, n_max + 1):
        for i in range(len(tokens) - n + 1):
            out.append(" ".join(tokens[i : i + n]))
    return out


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head - 20
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


@dataclass
class _Doc:
    doc_id: str
    vec: dict[str, float]
    payload: object


@dataclass
class TfidfIndex:
    """In-memory TF-IDF cosine-similarity index (fine for thousands of documents)."""

    docs: list[_Doc] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)
    _tf: list[Counter] = field(default_factory=list)
    _dirty: bool = False

    def add(self, doc_id: str, text: str, payload: object = None) -> None:
        tf = Counter(content_tokens(text))
        self._tf.append(tf)
        self._df.update(tf.keys())
        self.docs.append(_Doc(doc_id, {}, payload))
        self._dirty = True

    def __len__(self) -> int:
        return len(self.docs)

    def _idf(self, term: str) -> float:
        return math.log((1 + len(self.docs)) / (1 + self._df.get(term, 0))) + 1.0

    def _vectorize(self, tf: Counter) -> dict[str, float]:
        vec = {t: (1 + math.log(c)) * self._idf(t) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def _reindex(self) -> None:
        if self._dirty:
            for doc, tf in zip(self.docs, self._tf):
                doc.vec = self._vectorize(tf)
            self._dirty = False

    def query(self, text: str, k: int = 5, exclude: set[str] | None = None, min_sim: float = 0.0):
        """Return up to k (doc_id, similarity, payload) tuples, most similar first."""
        if not self.docs or k <= 0:
            return []
        self._reindex()
        q = self._vectorize(Counter(content_tokens(text)))
        if not q:
            return []
        scored = []
        for doc in self.docs:
            if exclude and doc.doc_id in exclude:
                continue
            sim = sum(w * doc.vec.get(t, 0.0) for t, w in q.items())
            if sim > min_sim:
                scored.append((doc.doc_id, sim, doc.payload))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
