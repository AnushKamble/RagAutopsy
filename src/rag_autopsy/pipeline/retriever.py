"""BM25 retrieval engine with candidate pool score logging."""

import re
import string

from rank_bm25 import BM25Okapi

from rag_autopsy.core.schemas import RetrievedPassage
from rag_autopsy.pipeline.corpus import Corpus


def simple_tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words with punctuation removed."""
    clean_text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text.lower())
    return [token for token in clean_text.split() if token]


class BM25Retriever:
    """Pure-Python BM25 retriever using rank-bm25."""

    def __init__(self, corpus: Corpus, k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.passages = corpus.passages

        # Pre-tokenize all passage texts
        self.corpus_tokens = [simple_tokenize(p.text) for p in self.passages]
        if self.corpus_tokens:
            self._bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
        else:
            self._bm25 = None

    def retrieve(
        self, query: str, top_k: int = 3, candidate_pool_size: int = 20
    ) -> tuple[list[RetrievedPassage], list[float]]:
        """Retrieve top-k passages and candidate pool scores.

        Returns:
            Tuple of:
            - top_k RetrievedPassage objects (rank 1 to top_k)
            - candidate_pool_scores list (floats of top candidate_pool_size scores)
        """
        if not self._bm25 or not self.passages:
            return [], []

        query_tokens = simple_tokenize(query)
        if not query_tokens:
            return [], []

        scores = self._bm25.get_scores(query_tokens)

        # Pair scores with passages and sort descending
        scored_pairs = sorted(
            zip(self.passages, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        candidate_pool_scores = [round(float(s), 4) for _, s in scored_pairs[:candidate_pool_size]]

        top_k_passages: list[RetrievedPassage] = []
        for rank_idx, (passage, score) in enumerate(scored_pairs[:top_k], start=1):
            top_k_passages.append(
                RetrievedPassage(
                    passage_id=passage.passage_id,
                    rank=rank_idx,
                    score=round(float(score), 4),
                    text=passage.text,
                    title=getattr(passage, "title", None),
                )
            )

        return top_k_passages, candidate_pool_scores


class DenseRetriever:
    """Dense semantic vector retriever using TF-IDF / sublinear n-gram embeddings and cosine similarity."""

    def __init__(self, corpus: Corpus, ngram_range: tuple[int, int] = (1, 2), min_df: int = 1):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.corpus = corpus
        self.passages = corpus.passages
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=min_df,
            token_pattern=r"(?u)\b\w+\b",
        )
        if self.passages:
            texts = [p.text for p in self.passages]
            self.doc_embeddings = self.vectorizer.fit_transform(texts)
        else:
            self.doc_embeddings = None

    def retrieve(
        self, query: str, top_k: int = 3, candidate_pool_size: int = 20
    ) -> tuple[list[RetrievedPassage], list[float]]:
        from sklearn.metrics.pairwise import cosine_similarity

        if self.doc_embeddings is None or not self.passages:
            return [], []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_embeddings)[0]

        scored_pairs = sorted(
            zip(self.passages, sims),
            key=lambda pair: pair[1],
            reverse=True,
        )

        candidate_pool_scores = [round(float(s), 4) for _, s in scored_pairs[:candidate_pool_size]]
        top_k_passages: list[RetrievedPassage] = []
        for rank_idx, (passage, score) in enumerate(scored_pairs[:top_k], start=1):
            top_k_passages.append(
                RetrievedPassage(
                    passage_id=passage.passage_id,
                    rank=rank_idx,
                    score=round(float(score), 4),
                    text=passage.text,
                    title=getattr(passage, "title", None),
                )
            )

        return top_k_passages, candidate_pool_scores


class HybridRetriever:
    """Hybrid retriever combining lexical BM25 and dense semantic scores via weighted score fusion."""

    def __init__(
        self,
        corpus: Corpus,
        alpha: float = 0.5,
        bm25_retriever: BM25Retriever | None = None,
        dense_retriever: DenseRetriever | None = None,
    ):
        self.corpus = corpus
        self.alpha = alpha  # Weight for BM25 (1 - alpha for Dense)
        self.bm25 = bm25_retriever or BM25Retriever(corpus)
        self.dense = dense_retriever or DenseRetriever(corpus)
        self.passages = corpus.passages

    def retrieve(
        self, query: str, top_k: int = 3, candidate_pool_size: int = 20
    ) -> tuple[list[RetrievedPassage], list[float]]:
        if not self.passages:
            return [], []

        # Get full scores from both retrievers
        bm25_top, bm25_scores = self.bm25.retrieve(query, top_k=len(self.passages), candidate_pool_size=len(self.passages))
        dense_top, dense_scores = self.dense.retrieve(query, top_k=len(self.passages), candidate_pool_size=len(self.passages))

        bm25_dict = {p.passage_id: p.score for p in bm25_top}
        dense_dict = {p.passage_id: p.score for p in dense_top}

        max_bm25 = max(bm25_dict.values()) if bm25_dict and max(bm25_dict.values()) > 0 else 1.0
        max_dense = max(dense_dict.values()) if dense_dict and max(dense_dict.values()) > 0 else 1.0

        hybrid_scored = []
        for p in self.passages:
            norm_b = bm25_dict.get(p.passage_id, 0.0) / max_bm25
            norm_d = dense_dict.get(p.passage_id, 0.0) / max_dense
            combined_score = self.alpha * norm_b + (1.0 - self.alpha) * norm_d
            hybrid_scored.append((p, combined_score))

        hybrid_scored.sort(key=lambda pair: pair[1], reverse=True)

        candidate_pool_scores = [round(float(s), 4) for _, s in hybrid_scored[:candidate_pool_size]]
        top_k_passages: list[RetrievedPassage] = []
        for rank_idx, (passage, score) in enumerate(hybrid_scored[:top_k], start=1):
            top_k_passages.append(
                RetrievedPassage(
                    passage_id=passage.passage_id,
                    rank=rank_idx,
                    score=round(float(score), 4),
                    text=passage.text,
                    title=getattr(passage, "title", None),
                )
            )

        return top_k_passages, candidate_pool_scores
