from __future__ import annotations

import os
from functools import lru_cache

from sentence_transformers import CrossEncoder
from dotenv import load_dotenv


load_dotenv()


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    model_name = os.getenv(
        "RERANKER_MODEL",
        DEFAULT_MODEL,
    )

    return CrossEncoder(model_name)


def rerank_results(
    query: str,
    results: list[dict],
    limit: int = 5,
) -> list[dict]:

    if not results:
        return []

    if len(results) <= 1:
        return results[:limit]

    reranker = get_reranker()

    pairs = [
        (
            query,
            result.get("content", ""),
        )
        for result in results
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    ranked_results = []

    for result, score in zip(
        results,
        scores,
    ):

        enriched_result = dict(result)

        enriched_result["rerank_score"] = float(
            score
        )

        ranked_results.append(
            enriched_result
        )

    ranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return ranked_results[:limit]