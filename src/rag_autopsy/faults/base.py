"""Base interfaces and result containers for fault injectors."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InjectionResult(BaseModel):
    """Result of attempting a controlled fault injection on a trial."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    injected_fault: str
    fault_parameters: dict[str, Any] = Field(default_factory=dict)
    injection_applied: bool
    trace_condition_achieved: bool
    trial_valid: bool
    invalid_reason: str | None = None
    modified_corpus: Any | None = None


class BaseFaultInjector(ABC):
    """Abstract base class for controlled fault injectors."""

    @abstractmethod
    def inject(self, qa_record: Any, base_corpus: Any) -> InjectionResult:
        """Apply the controlled intervention and return the injection result."""
