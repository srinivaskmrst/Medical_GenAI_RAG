"""Google embedding provider backed by the Gemini embeddings API."""

from __future__ import annotations

from google import genai
from google.genai.errors import ClientError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

from app.config.settings import get_settings
from app.embeddings.base import EmbeddingProvider

DEFAULT_DIMENSION = 3072
# The embed_content API rejects batches over 100 requests, and the free tier
# also caps overall throughput at ~100 embed requests/minute — one full
# batch can exhaust the whole per-minute quota.
MAX_BATCH_SIZE = 100


def _is_rate_limited(exc: BaseException) -> bool:
    return isinstance(exc, ClientError) and exc.code == 429


@retry(retry=retry_if_exception(_is_rate_limited), wait=wait_fixed(60), stop=stop_after_attempt(5))
def _embed_content(client: genai.Client, model_name: str, contents: object):
    return client.models.embed_content(model=model_name, contents=contents)


class GoogleEmbedding(EmbeddingProvider):
    """Embedding provider using Google's Gemini embedding models."""

    def __init__(self, model_name: str = "gemini-embedding-001", api_key: str | None = None, **_: object) -> None:
        self.model_name = model_name
        resolved_key = api_key or get_settings().google_api_key
        if not resolved_key:
            raise ValueError("GOOGLE_API_KEY is not set")
        self.client = genai.Client(api_key=resolved_key)
        self.dimension: int | None = None

    def _validate_dimension(self, values: list[float]) -> list[float]:
        if self.dimension is None:
            self.dimension = len(values)
        elif len(values) != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, received {len(values)}"
            )
        return values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for start in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[start : start + MAX_BATCH_SIZE]
            response = _embed_content(self.client, self.model_name, batch)
            results.extend(self._validate_dimension(list(embedding.values)) for embedding in response.embeddings)
        return results

    def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise ValueError("Query must not be empty")
        response = _embed_content(self.client, self.model_name, query)
        return self._validate_dimension(list(response.embeddings[0].values))

    def get_dimension(self) -> int:
        # gemini-embedding-001 always outputs DEFAULT_DIMENSION unless
        # output_dimensionality truncation is requested (not used here).
        return DEFAULT_DIMENSION
