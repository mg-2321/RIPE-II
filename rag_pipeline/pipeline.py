"""
End-to-end RAG pipeline scaffold inspired by RAG'n'Roll.

Author: Gayatri Malladi
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from .chunking import Chunker, ChunkerConfig
from .document_store import DocumentStore
from .query_processing import apply_processors, build_processors
from prompting import build_pipeline_prompt_prefix, infer_domain_from_identifier

# guards/ and retrievers/ are top-level packages (siblings of rag_pipeline_components/)
#
# Some local evaluation jobs are run from partially materialized worktrees where the
# optional `guards` package is not present. Fall back to no-op guardrails so the
# retrieval/generation pipeline still works for raw attack-evaluation runs.
try:
    from guards import GuardDecision, Guardrail, DEFAULT_GUARDS, build_guards
except ImportError:  # pragma: no cover - environment-dependent fallback
    @dataclass
    class GuardDecision:
        allow: bool
        reason: str = ""
        fallback: Optional[str] = None

        @staticmethod
        def allow_pass() -> "GuardDecision":
            return GuardDecision(True, "")

    class Guardrail:
        def inspect_retrieved(self, documents) -> GuardDecision:
            return GuardDecision.allow_pass()

        def inspect_prompt(self, prompt: str) -> GuardDecision:
            return GuardDecision.allow_pass()

        def inspect_generation(self, prompt: str, answer: str) -> GuardDecision:
            return GuardDecision.allow_pass()

    DEFAULT_GUARDS: List[Guardrail] = []

    def build_guards(_guard_names: List[str]) -> List[Guardrail]:
        return []
from retrievers import BaseRetriever, get_retriever

if TYPE_CHECKING:  # pragma: no cover
    # Avoid importing transformers-heavy generator module unless generation is enabled.
    from .generator import GenerationConfig
    from .rerankers import BaseReranker


@dataclass
class PipelineConfig:
    document_path: str
    retriever: str = "bm25"
    retriever_kwargs: Dict = field(default_factory=dict)
    candidate_pool_size: int = 12
    default_top_k: int = 6
    prompt_profile: str = "defensive"
    prompt_domain: str = "auto"
    generation: Optional["GenerationConfig"] = None
    guards: Optional[List[Guardrail]] = None
    guard_names: List[str] = field(default_factory=list)
    chunker: Optional[ChunkerConfig] = None
    query_processors: List[str] = field(default_factory=list)
    reranker: Optional[str] = None
    reranker_kwargs: Dict = field(default_factory=dict)
    shared_generator: Optional[object] = None
    document_store_mode: str = "auto"


class Pipeline:
    """
    High-level RAG orchestrator.  Responsible for wiring:
      - Document store
      - Retriever
      - Guardrails (pre/post)
      - Generator
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        chunker = Chunker(config.chunker) if config.chunker else None
        store_mode = (config.document_store_mode or "auto").strip().lower()
        if store_mode not in {"auto", "memory", "lazy"}:
            raise ValueError(
                f"Unknown document_store_mode '{config.document_store_mode}'. "
                "Available: auto, memory, lazy"
            )
        if store_mode == "auto":
            doc_path = Path(config.document_path)
            store_mode = "lazy" if chunker is None and doc_path.exists() and doc_path.stat().st_size >= 250_000_000 else "memory"
        self.store = DocumentStore.from_jsonl(
            config.document_path,
            chunker=chunker,
            lazy=(store_mode == "lazy"),
        )
        domain_identifier = (
            config.document_path
            if (config.prompt_domain or "auto").strip().lower() == "auto"
            else config.prompt_domain
        )
        self.prompt_domain = infer_domain_from_identifier(domain_identifier)
        retriever_cls = get_retriever(config.retriever)
        self.retriever: BaseRetriever = retriever_cls(self.store, **config.retriever_kwargs)
        if config.shared_generator is not None:
            self.generator = config.shared_generator
        elif config.generation:
            # Lazy import to keep retrieval-only runs fast and robust.
            from .generator import Generator

            self.generator = Generator(config.generation)
        else:
            self.generator = None
        # Distinguish "explicitly no guards" ([]) from "use defaults" (None).
        if config.guards is not None:
            self.guards = config.guards
        elif config.guard_names:
            self.guards = build_guards(config.guard_names)
        else:
            self.guards = DEFAULT_GUARDS
        self.processors = build_processors(config.query_processors)
        self.reranker: Optional["BaseReranker"] = None
        if config.reranker:
            from .rerankers import get_reranker

            self.reranker = get_reranker(config.reranker, **config.reranker_kwargs)

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        retrieval = self.retrieve_context(query, top_k=top_k)
        if not retrieval["guard_decision"]["allow"]:
            return self._build_response(
                retrieval["original_query"],
                retrieval["processed_query"],
                retrieval["_raw_retrieved"],
                "",
                self._guard_decision_from_dict(retrieval["guard_decision"]),
            )

        generation = self.answer_from_context(
            retrieval["processed_query"],
            retrieval["context"],
            conversation_history=conversation_history,
        )
        if not generation["guard_decision"]["allow"]:
            return self._build_response(
                retrieval["original_query"],
                retrieval["processed_query"],
                retrieval["_raw_retrieved"],
                generation["answer"],
                self._guard_decision_from_dict(generation["guard_decision"]),
            )

        return self._build_response(
            retrieval["original_query"],
            retrieval["processed_query"],
            retrieval["_raw_retrieved"],
            generation["answer"],
            GuardDecision.allow_pass(),
        )

    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> Dict:
        original_query = query
        processed_query = apply_processors(query, self.processors) if self.processors else query
        final_top_k = top_k or self.config.default_top_k

        candidate_pool = max(final_top_k, self.config.candidate_pool_size)
        retrieved = self.retriever.retrieve(processed_query, top_k=candidate_pool)

        if self.reranker:
            retrieved = self.reranker.rerank(processed_query, retrieved)

        retrieved = retrieved[:final_top_k]
        documents = [doc for doc, _ in retrieved]

        for guard in self.guards:
            decision = guard.inspect_retrieved(documents)
            if not decision.allow:
                return {
                    "original_query": original_query,
                    "processed_query": processed_query,
                    "retrieved": self._serialize_retrieved(retrieved),
                    "documents": self._serialize_documents(documents),
                    "context": "",
                    "guard_decision": {
                        "allow": decision.allow,
                        "reason": decision.reason,
                        "fallback": decision.fallback,
                    },
                    "_raw_retrieved": retrieved,
                }

        aggregated_context = self._format_context(documents)
        return {
            "original_query": original_query,
            "processed_query": processed_query,
            "retrieved": self._serialize_retrieved(retrieved),
            "documents": self._serialize_documents(documents),
            "context": aggregated_context,
            "guard_decision": {"allow": True, "reason": ""},
            "_raw_retrieved": retrieved,
        }

    def answer_from_context(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        prompt = self._build_prompt(
            query,
            context,
            conversation_history=conversation_history,
        )

        for guard in self.guards:
            decision = guard.inspect_prompt(prompt)
            if not decision.allow:
                return {
                    "answer": decision.fallback or "",
                    "prompt": prompt,
                    "guard_decision": {
                        "allow": decision.allow,
                        "reason": decision.reason,
                        "fallback": decision.fallback,
                    },
                }

        answer = ""
        if self.generator:
            answer = self.generator.generate(prompt)
            for guard in self.guards:
                decision = guard.inspect_generation(prompt, answer)
                if not decision.allow:
                    return {
                        "answer": decision.fallback or "",
                        "prompt": prompt,
                        "guard_decision": {
                            "allow": decision.allow,
                            "reason": decision.reason,
                            "fallback": decision.fallback,
                        },
                    }

        return {
            "answer": answer,
            "prompt": prompt,
            "guard_decision": {"allow": True, "reason": ""},
        }

    def _format_context(self, documents: Iterable) -> str:
        """Format retrieved documents into a plain context string.

        No [POISONED]/[CLEAN] labels are added — those would let the LLM trivially
        ignore injected documents and would make ASR measurements meaningless.
        All retrieved documents are included in full, in retrieval rank order.
        """
        formatted = []
        for doc in documents:
            part = f"{doc.title}\n{doc.text}" if getattr(doc, 'title', '') else doc.text
            formatted.append(part)
        return "\n\n".join(formatted)

    @staticmethod
    def _format_history(conversation_history: Optional[List[Dict[str, str]]]) -> str:
        if not conversation_history:
            return ""

        turns: List[str] = []
        for item in conversation_history:
            role = str(item.get("role", "user")).strip().upper() or "USER"
            content = str(item.get("content", "")).strip()
            if content:
                turns.append(f"{role}: {content}")

        if not turns:
            return ""

        return "Conversation so far:\n" + "\n".join(turns) + "\n\n"

    def _build_prompt(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        history_block = self._format_history(conversation_history)
        prompt_prefix = self._PROMPT_PROFILES.get(
            self.config.prompt_profile,
            "defensive",
        )
        return (
            build_pipeline_prompt_prefix(prompt_prefix, self.prompt_domain)
            + f"{history_block}"
            + f"Context:\n{context}\n\n"
            + f"User question: {query}\nAnswer:"
        )

    def _serialize_retrieved(self, retrieved) -> List[Dict]:
        return [
            {"doc_id": doc.doc_id, "score": score, "poisoned": doc.is_poisoned}
            for doc, score in retrieved
        ]

    @staticmethod
    def _serialize_documents(documents: Iterable) -> List[Dict]:
        return [
            {
                "doc_id": doc.doc_id,
                "title": getattr(doc, "title", ""),
                "text": getattr(doc, "text", ""),
                "poisoned": getattr(doc, "is_poisoned", False),
            }
            for doc in documents
        ]

    @staticmethod
    def _guard_decision_from_dict(payload: Dict) -> GuardDecision:
        return GuardDecision(
            allow=bool(payload.get("allow", False)),
            reason=str(payload.get("reason", "")),
            fallback=payload.get("fallback"),
        )

    @staticmethod
    def _build_response(original_query: str, processed_query: str, retrieved, answer: str, decision: GuardDecision) -> Dict:
        return {
            "original_query": original_query,
            "processed_query": processed_query,
            "answer": answer,
            "retrieved": [
                {"doc_id": doc.doc_id, "score": score, "poisoned": doc.is_poisoned}
                for doc, score in retrieved
            ],
            "guard_decision": {
                "allow": decision.allow,
                "reason": decision.reason,
                "fallback": decision.fallback,
            },
        }
    _PROMPT_PROFILES = {
        "defensive": "defensive",
        "attack_eval": "attack_eval",
    }
