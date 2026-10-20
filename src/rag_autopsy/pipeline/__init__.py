"""RAG reference pipeline components: corpus, retriever, context builder, generator."""

from rag_autopsy.pipeline.context_builder import ContextBuilder
from rag_autopsy.pipeline.corpus import Corpus
from rag_autopsy.pipeline.generator import (
    BaseGenerator,
    CachedGenerator,
    MockExtractiveGenerator,
    OllamaGenerator,
    OpenAICompatibleGenerator,
)
from rag_autopsy.pipeline.rag_pipeline import RAGPipeline
from rag_autopsy.pipeline.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
)

__all__ = [
    "BM25Retriever",
    "BaseGenerator",
    "CachedGenerator",
    "ContextBuilder",
    "Corpus",
    "DenseRetriever",
    "HybridRetriever",
    "MockExtractiveGenerator",
    "OllamaGenerator",
    "OpenAICompatibleGenerator",
    "RAGPipeline",
]
