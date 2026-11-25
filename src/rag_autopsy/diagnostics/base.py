"""Base diagnostic interface enforcing strict visible-trace input constraints."""

from abc import ABC, abstractmethod

from rag_autopsy.core.schemas import FORBIDDEN_LEAKAGE_KEYS, Diagnosis, DiagnosticTrace


class BaseDiagnostic(ABC):
    """Abstract base class for all diagnostic methods.

    Guarantees that methods can only receive and inspect `DiagnosticTrace`.
    Rejects convenience wrappers, combined records, and objects smuggling privileged fields.
    """

    def __init__(self, name: str):
        self.name = name

    def diagnose(self, trace: DiagnosticTrace) -> Diagnosis:
        """Entrypoint verifying input type and strict boundary constraints."""
        if not isinstance(trace, DiagnosticTrace) or type(trace) is not DiagnosticTrace:
            raise TypeError(
                f"Data boundary violation: Diagnostic '{self.name}' must only receive DiagnosticTrace, "
                f"got {type(trace).__name__}!"
            )
        for forbidden in FORBIDDEN_LEAKAGE_KEYS:
            if hasattr(trace, forbidden):
                raise ValueError(
                    f"Data boundary violation: trace object contains forbidden attribute '{forbidden}'!"
                )
        return self._predict(trace)


    @abstractmethod
    def _predict(self, trace: DiagnosticTrace) -> Diagnosis:
        """Internal diagnostic logic implemented by subclasses."""
