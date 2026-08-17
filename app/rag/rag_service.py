"""Central orchestration for the RAG pipeline."""

from __future__ import annotations

from typing import Any

from app.config.settings import get_settings
from app.embeddings import get_embedding_provider
from app.guardrails.input import InputGuardrail
from app.guardrails.output import OutputGuardrail
from app.guardrails.safety import SafetyGuardrail
from app.llm.factory import get_llm_provider
from app.rag.context_builder import ContextBuilder
from app.rag.grounding import GroundingChecker
from app.rag.long_term_memory import LongTermMemory
from app.rag.prompt_manager import PromptManager
from app.rag.query_encoder import QueryEncoder
from app.rag.semantic_cache import SemanticCache
from app.retrieval.filters import AuthorizationFilter
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.keyword import BM25Retriever, get_keyword_retriever
from app.retrieval.reranker import Reranker
from app.retrieval.semantic import SemanticRetriever

GUARDRAIL_BLOCKED_MESSAGE = (
    "This response was withheld by a safety guardrail. Please consult a qualified clinician."
)


class RAGService:
    """Orchestrates the full retrieval-augmented generation flow."""

    def __init__(
        self,
        query_encoder: QueryEncoder | None = None,
        semantic_retriever: SemanticRetriever | None = None,
        keyword_retriever: BM25Retriever | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        context_builder: ContextBuilder | None = None,
        prompt_manager: PromptManager | None = None,
        llm_provider: Any | None = None,
        reranker: Reranker | None = None,
        semantic_cache: SemanticCache | None = None,
        long_term_memory: LongTermMemory | None = None,
        input_guardrail: InputGuardrail | None = None,
        safety_guardrail: SafetyGuardrail | None = None,
        output_guardrail: OutputGuardrail | None = None,
        authorization_filter: AuthorizationFilter | None = None,
        grounding_checker: GroundingChecker | None = None,
    ) -> None:
        settings = get_settings()
        embedding_provider = get_embedding_provider()

        self.query_encoder = query_encoder or QueryEncoder(embedding_provider)
        self.semantic_retriever = semantic_retriever or SemanticRetriever(
            top_k=settings.top_k, embedding_provider=embedding_provider
        )
        self.keyword_retriever = keyword_retriever or get_keyword_retriever()
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(
            semantic_retriever=self.semantic_retriever,
            keyword_retriever=self.keyword_retriever,
            top_k=settings.top_k,
        )
        self.context_builder = context_builder or ContextBuilder(max_context_tokens=2000)
        self.prompt_manager = prompt_manager or PromptManager()
        self.llm_provider = llm_provider or get_llm_provider()
        self.reranker = reranker or Reranker()
        self.semantic_cache = semantic_cache or SemanticCache()
        self.long_term_memory = long_term_memory or LongTermMemory()
        self.input_guardrail = input_guardrail or InputGuardrail()
        self.safety_guardrail = safety_guardrail or SafetyGuardrail()
        self.output_guardrail = output_guardrail or OutputGuardrail()
        self.authorization_filter = authorization_filter or AuthorizationFilter()
        self.grounding_checker = grounding_checker or GroundingChecker(
            min_confidence=settings.grounding_min_confidence
        )
        self.rerank_top_k = settings.rerank_top_k

    def answer(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        query = self.query_encoder.normalize(query)
        self.safety_guardrail.validate_question(query)
        self.input_guardrail.validate(query)

        query_vector = self.query_encoder.encode(query)

        cached = self.semantic_cache.lookup(query_vector)
        if cached is not None:
            return cached

        long_term_context = self.long_term_memory.retrieve(session_id, query_vector)

        results = self.hybrid_retriever.retrieve(query=query, filters=filters, top_k=top_k)
        results = self.authorization_filter.apply(results)
        results = self.reranker.rerank(query, results)
        if self.rerank_top_k:
            results = results[: self.rerank_top_k]
        context_items = self.context_builder.build(results)

        context_sections = []
        if long_term_context:
            prior_text = "\n".join(f"[{item.get('role', 'user')}] {item.get('text', '')}" for item in long_term_context)
            context_sections.append(f"PRIOR CONVERSATION CONTEXT (this session)\n{prior_text}")
        if context_items:
            context_sections.append(
                "\n\n".join(
                    f"Document: {item.get('document_id', 'unknown')} | Source: {item.get('source', 'unknown')} | Page: {item.get('page_number', 'n/a')}\n{item.get('text', '')}"
                    for item in context_items
                )
            )

        if context_sections:
            prompt = self.prompt_manager.build_qa_prompt(query, "\n\n".join(context_sections))
        else:
            prompt = self.prompt_manager.build_insufficient_context_prompt(query)

        answer = self.llm_provider.generate(prompt)

        guardrail_blocked = False
        try:
            self.output_guardrail.validate(answer)
            self.safety_guardrail.validate_answer(answer)
        except ValueError:
            answer = GUARDRAIL_BLOCKED_MESSAGE
            guardrail_blocked = True

        confidence = self.grounding_checker.check(answer, context_items)

        sources = [
            {
                "document_id": item.get("document_id"),
                "document_name": item.get("document_name") or item.get("source"),
                "page_number": item.get("page_number"),
                "chunk_id": item.get("chunk_id"),
                "score": item.get("score"),
            }
            for item in context_items
        ]
        retrieval_metadata = {
            "top_k": top_k,
            "num_results": len(results),
            "context_items": len(context_items),
            "cache_hit": False,
            "guardrail_blocked": guardrail_blocked,
            "long_term_context_items": len(long_term_context),
        }
        usage = {"tokens": len(answer.split())}

        if not guardrail_blocked:
            self.semantic_cache.store(query, query_vector, answer, sources, retrieval_metadata, usage, confidence)
            self.long_term_memory.store(session_id, "user", query, query_vector)
            answer_vector = self.query_encoder.encode(answer) if answer.strip() else query_vector
            self.long_term_memory.store(session_id, "assistant", answer, answer_vector)

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_metadata": retrieval_metadata,
            "usage": usage,
            "confidence": confidence,
        }
