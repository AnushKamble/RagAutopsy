# Generator Adapter Feasibility & Laptop Fit Analysis

This document provides a formal evaluation of generator options for RAG-Autopsy under the core constraints:
1. **Student Laptop Constraint:** Must run locally on CPU/low-memory environments (16GB RAM, CPU-first, fast iteration).
2. **Offline Reproducibility:** No mandatory external API keys or recurring cloud costs.
3. **Licensing & Terms:** Permissive open licenses (Apache 2.0, MIT) allowing non-commercial and commercial evaluation.
4. **Caching Invariant:** Strict SHA-256 caching across all result-changing inputs (model identifier, generation parameters, question, assembled context).

---

## 1. Candidate Comparison Matrix

| Option | Architecture / Model | Parameter Count | Disk & RAM Footprint | License / Terms | Laptop Feasibility (CPU) | Recommendation Status |
|---|---|---|---|---|---|---|
| **Deterministic Test Heuristic** | Regex/Overlap Extractor (`MockExtractiveGenerator`) | 0 | 0 MB / 0 MB | MIT (Built-in) | Instant (< 1ms/query) | **Approved for Infrastructure & Fault Validation Tests** (strictly disclosed as non-generative) |
| **Flan-T5-Small** | Encoder-Decoder (`google/flan-t5-small`) | 77M | ~308 MB / < 1 GB | Apache 2.0 | High (~80–150ms/query on CPU) | **Selected Primary Local Generative Adapter** |
| **Flan-T5-Base** | Encoder-Decoder (`google/flan-t5-base`) | 248M | ~990 MB / < 2 GB | Apache 2.0 | Moderate (~250–500ms/query on CPU) | **Approved Secondary Generative Option** |
| **Qwen2.5-0.5B-Instruct** | Decoder-Only (`Qwen/Qwen2.5-0.5B-Instruct`) | 490M | ~1.0 GB / ~2 GB | Apache 2.0 | Moderate (~300–700ms/query on CPU) | **Approved Optional Instruction Adapter** |
| **Local Daemon (Ollama / vLLM)** | GGUF / OpenAI-compatible endpoint (`OpenAICompatibleGenerator`) | Variable (e.g. 1B–3B) | Host-managed | Variable (e.g. Llama 3.2, Qwen) | Offloaded to local daemon | **Supported via `OpenAICompatibleGenerator`** |
| **Proprietary Cloud APIs** | OpenAI / Anthropic / Gemini | N/A | Cloud | Commercial terms, API billing | Requires internet, API keys, and recurring cost | **Rejected as mandatory core dependency** (optional external proxy only) |

---

## 2. Model & License Terms Verification

1. **`google/flan-t5-small` and `google/flan-t5-base`:**
   - **Provider:** Google Research.
   - **License:** Apache License 2.0 (permissive, open for research and commercial distribution).
   - **Task Alignment:** Pre-trained specifically on academic question answering and span extraction datasets.
   - **Hardware Fit:** 77M parameters require only ~308 MB of RAM in FP32, executing with near-zero latency on modern laptop CPUs (e.g. AMD Ryzen 9 8940HX).

2. **OpenAI-Compatible Local Endpoint (`OpenAICompatibleGenerator`):**
   - Allows users with local Ollama (`ollama run qwen2.5:0.5b`) or LM Studio installations to connect zero-dependency via Python standard library `urllib.request`.
   - Leaves host machine memory management to the external runtime.

---

## 3. Implemented Adapter Contracts

Both adapters inherit from `BaseGenerator` in `src/rag_autopsy/pipeline/generator.py`:
- `get_model_name()`: Returns unique adapter key (e.g., `hf_google_flan-t5-small`, `api_qwen2.5_0.5b`).
- `get_config()`: Returns canonical dictionary of temperature, max tokens, endpoint, and device settings.
- `CachedGenerator`: SHA-256 hashes `model_name`, `get_config()`, `question`, and `context`, ensuring that no cached entry is re-used if prompt templates or generator parameters change.
