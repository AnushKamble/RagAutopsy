"""BM25 retrieval engine with candidate pool score logging."""

import re
import string
from typing import List, Tuple
from rank_bm25 import BM25Okapi

from rag_autopsy.core.schemas import RetrievedPassage
from rag_autopsy.pipeline.corpus import Corpus


def simple_tokenize(text: str) -> List[str]:
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
    ) -> Tuple[List[RetrievedPassage], List[float]]:
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

        top_k_passages: List[RetrievedPassage] = []
        for rank_idx, (passage, score) in enumerate(scored_pairs[:top_k], start=1):
            top_k_passages.append(
                RetrievedPassage(
                    passage_id=passage.passage_id,
                    rank=rank_idx,
                    score=round(float(score), 4),
                    text=passage.text,
                )
            )

        return top_k_passages, candidate_pool_scores
