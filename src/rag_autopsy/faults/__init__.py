"""Fault injection engines for RAG-Autopsy (F0–F4)."""

from rag_autopsy.faults.base import BaseFaultInjector, InjectionResult
from rag_autopsy.faults.f1_unavailable import F1EvidenceUnavailableInjector
from rag_autopsy.faults.f2_ranked_out import F2EvidenceRankedOutInjector
from rag_autopsy.faults.f3_assembly_loss import F3ContextAssemblyLossInjector
from rag_autopsy.faults.f4_contradiction import F4ContradictoryEvidenceInjector
from rag_autopsy.faults.f5_unsupported import F5UnsupportedGenerationInjector
from rag_autopsy.faults.combined import CombinedFaultInjector

__all__ = [
    "BaseFaultInjector",
    "InjectionResult",
    "F1EvidenceUnavailableInjector",
    "F2EvidenceRankedOutInjector",
    "F3ContextAssemblyLossInjector",
    "F4ContradictoryEvidenceInjector",
    "F5UnsupportedGenerationInjector",
    "CombinedFaultInjector",
]
