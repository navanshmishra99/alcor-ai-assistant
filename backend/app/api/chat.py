from __future__ import annotations

import time

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.app.services.rag import build_context
from backend.app.services.ai import generate_answer
from backend.app.services.grounding import validate_answer


router = APIRouter()


# ---------------------------------------------------------
# Conversation models
# ---------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


# ---------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    total_start = time.perf_counter()

    current_message = request.message.strip()

    # -----------------------------------------------------
    # Empty message
    # -----------------------------------------------------

    if not current_message:

        return ChatResponse(
            answer="How can I help you today?",
            sources=[],
        )

    # -----------------------------------------------------
    # Conversation history
    # -----------------------------------------------------
    #
    # Keep recent conversation available to the AI so it
    # can understand natural follow-ups such as:
    #
    # "Who is the CEO?"
    # "Where is he based?"
    # "Tell me more about him."
    #
    # The history itself is not displayed to the user.
    #

    conversation = request.history[-10:]

    history_for_ai = []

    for message in conversation:

        role = message.role.strip().lower()
        content = message.content.strip()

        if role not in {"user", "assistant"}:
            continue

        if not content:
            continue

        history_for_ai.append(
            {
                "role": role,
                "content": content,
            }
        )

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------
    #
    # Use the current question together with recent USER
    # messages.
    #
    # Assistant answers are deliberately not included in the
    # retrieval query because they can introduce unrelated
    # words and cause irrelevant knowledge chunks to rank
    # highly.
    #
    # Conversation history is still passed separately to the
    # LLM for natural conversation and follow-up resolution.
    #

    retrieval_parts = []

    for message in reversed(history_for_ai):

        if message["role"] != "user":
            continue

        retrieval_parts.append(
            message["content"]
        )

        # Keep retrieval focused on recent user intent.
        if len(retrieval_parts) >= 3:
            break

    retrieval_parts.reverse()

    retrieval_parts.append(
        current_message
    )

    retrieval_query = "\n".join(
        retrieval_parts
    )

    # -----------------------------------------------------
    # RAG
    # -----------------------------------------------------

    rag_start = time.perf_counter()

    rag_result = build_context(
        retrieval_query
    )

    rag_time = (
        time.perf_counter()
        - rag_start
    )

    context = rag_result["context"]

    # -----------------------------------------------------
    # AI generation
    # -----------------------------------------------------

    ai_start = time.perf_counter()

    answer = generate_answer(
        question=current_message,
        context=context,
        history=history_for_ai,
    )

    ai_time = (
        time.perf_counter()
        - ai_start
    )

    # -----------------------------------------------------
    # Source handling
    # -----------------------------------------------------
    #
    # Retrieval can return chunks even for ordinary
    # conversation. That does NOT mean those sources
    # support the response.
    #
    # Only expose sources when the generated answer is
    # actually supported by the retrieved knowledge.
    #

    sources = []

    if context and answer:

        try:

            validated_answer = validate_answer(
                answer=answer,
                context=context,
            )

            if (
                validated_answer
                and validated_answer.strip() == answer.strip()
            ):
                sources = rag_result["sources"]

        except Exception:

            sources = []

    # -----------------------------------------------------
    # Total timing
    # -----------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start
    )

    print(
        "\n========== ASK ALCOR TIMING =========="
    )

    print(
        f"History Messages: {len(history_for_ai)}"
    )

    print(
        f"Retrieval Query : {retrieval_query[:200]!r}"
    )

    print(
        f"RAG / Retrieval : {rag_time:.2f}s"
    )

    print(
        f"LLM Generation  : {ai_time:.2f}s"
    )

    print(
        f"Total           : {total_time:.2f}s"
    )

    print(
        f"Sources         : {len(sources)}"
    )

    print(
        "======================================\n"
    )

    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    return ChatResponse(
        answer=answer,
        sources=sources,
    )