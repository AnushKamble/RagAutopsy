"""Base diagnostic interface enforcing strict visible-trace input constraints."""

from abc import ABC, abstractmethod
from rag_autopsy.core.schemas import Diagnosis, DiagnosticTrace


class BaseDiagnostic(ABC):
    """Abstract base class for all diagnostic methods.
    
    Guarantees that methods can only receive and inspect `DiagnosticTrace`.
    """

    def __init__(self, name: str):
        self.name = name

    def diagnose(self, trace: DiagnosticTrace) -> Diagnosis:
        """Entrypoint verifying input type before delegating to internal prediction."""
        if not isinstance(trace, DiagnosticTrace):
            raise TypeError(
                f"Data boundary violation: Diagnostic '{self.name}' must only receive DiagnosticTrace, "
                f"got {type(trace).__name__}!"
            )
        return self._predict(trace)

    @abstractmethod
    def _predict(self, trace: DiagnosticTrace) -> Diagnosis:
        """Internal diagnostic logic implemented by subclasses."""
        pass
