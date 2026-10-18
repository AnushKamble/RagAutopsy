"""Diagnostics and attribution policy engine for RAG-Autopsy."""

from rag_autopsy.diagnostics.base import BaseDiagnostic
from rag_autopsy.diagnostics.baselines import MajorityCauseBaseline, RandomBaseline
from rag_autopsy.diagnostics.d1_retrieval import D1RetrievalDiagnostic
from rag_autopsy.diagnostics.d2_support import D2LexicalSupportDiagnostic
from rag_autopsy.diagnostics.d3_nli import D3NLISupportDiagnostic
from rag_autopsy.diagnostics.d4_learned import D4LearnedClassifierPolicy
from rag_autopsy.diagnostics.d4_policy import D4CombinedAttributionPolicy

__all__ = [
    "BaseDiagnostic",
    "MajorityCauseBaseline",
    "RandomBaseline",
    "D1RetrievalDiagnostic",
    "D2LexicalSupportDiagnostic",
    "D3NLISupportDiagnostic",
    "D4CombinedAttributionPolicy",
    "D4LearnedClassifierPolicy",
]
