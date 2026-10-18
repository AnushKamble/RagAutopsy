"""Context assembly and truncation logging engine."""

from typing import List, Tuple
from rag_autopsy.core.schemas import RetrievedPassage


class ContextBuilder:
    """Assembles retrieved passages into prompt context and tracks passage survival."""

    def __init__(self, max_context_chars: int = 4000, separator: str = "\n\n---\n\n"):
        self.max_context_chars = max_context_chars
        self.separator = separator

    def build_context(
        self, retrieved_passages: List[RetrievedPassage]
    ) -> Tuple[str, List[str], List[str]]:
        """Assemble passages into generator context, logging pre- and post-assembly passage IDs.
        
        Returns:
            Tuple of:
            - assembled_context: str
            - pre_assembly_passage_ids: List[str]
            - post_assembly_passage_ids: List[str]
        """
        pre_assembly_ids = [p.passage_id for p in retrieved_passages]
        post_assembly_ids: List[str] = []
        context_parts: List[str] = []

        current_length = 0
        for passage in retrieved_passages:
            formatted_passage = f"[Passage {passage.rank}]:\n{passage.text}"
            added_length = len(formatted_passage) + (len(self.separator) if context_parts else 0)

            if current_length + added_length <= self.max_context_chars:
                context_parts.append(formatted_passage)
                post_assembly_ids.append(passage.passage_id)
                current_length += added_length
            else:
                # Truncation: passage omitted due to budget constraint
                break

        assembled_context = self.separator.join(context_parts)
        return assembled_context, pre_assembly_ids, post_assembly_ids
