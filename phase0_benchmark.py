import os
import time
import json
import urllib.request
from dotenv import load_dotenv


load_dotenv()


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
).rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b",
)

OLLAMA_TIMEOUT_SECONDS = int(
    os.getenv("OLLAMA_BENCHMARK_TIMEOUT_SECONDS", "180")
)

OLLAMA_TEMPERATURE = float(
    os.getenv("OLLAMA_BENCHMARK_TEMPERATURE", "0")
)

OLLAMA_NUM_PREDICT = int(
    os.getenv("OLLAMA_BENCHMARK_NUM_PREDICT", "32")
)

OLLAMA_KEEP_ALIVE = os.getenv(
    "OLLAMA_BENCHMARK_KEEP_ALIVE",
    "10m",
)

BENCHMARK_PROMPT = os.getenv(
    "OLLAMA_BENCHMARK_PROMPT",
    "Alcor Solutions is a technology and consulting company. " * 24,
)

OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"


def build_payload() -> dict:
    return {
        "model": OLLAMA_MODEL,
        "prompt": BENCHMARK_PROMPT,
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }


def call_ollama() -> dict:
    payload = build_payload()

    request = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=OLLAMA_TIMEOUT_SECONDS,
    ) as response:
        return json.loads(
            response.read().decode("utf-8")
        )


def safe_rate(tokens: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0

    return tokens / seconds


def format_result(
    label: str,
    data: dict,
    elapsed: float,
) -> None:
    prompt_eval_seconds = (
        data.get("prompt_eval_duration", 0)
        / 1_000_000_000
    )

    generation_seconds = (
        data.get("eval_duration", 0)
        / 1_000_000_000
    )

    prompt_tokens = data.get(
        "prompt_eval_count",
        0,
    )

    output_tokens = data.get(
        "eval_count",
        0,
    )

    prompt_tokens_per_second = safe_rate(
        prompt_tokens,
        prompt_eval_seconds,
    )

    generation_tokens_per_second = safe_rate(
        output_tokens,
        generation_seconds,
    )

    print(f"{label}_total_seconds={elapsed:.3f}")
    print(
        f"{label}_prompt_eval_seconds="
        f"{prompt_eval_seconds:.3f}"
    )
    print(
        f"{label}_prompt_eval_tokens_per_sec="
        f"{prompt_tokens_per_second:.2f}"
    )
    print(
        f"{label}_generation_seconds="
        f"{generation_seconds:.3f}"
    )
    print(
        f"{label}_generation_tokens_per_sec="
        f"{generation_tokens_per_second:.2f}"
    )
    print(
        f"{label}_prompt_tokens="
        f"{prompt_tokens}"
    )
    print(
        f"{label}_output_tokens="
        f"{output_tokens}"
    )


def print_configuration() -> None:
    print("=" * 60)
    print("OLLAMA BENCHMARK CONFIGURATION")
    print("=" * 60)
    print(f"Model: {OLLAMA_MODEL}")
    print(f"URL: {OLLAMA_GENERATE_URL}")
    print(f"Temperature: {OLLAMA_TEMPERATURE}")
    print(f"Output tokens: {OLLAMA_NUM_PREDICT}")
    print(f"Keep alive: {OLLAMA_KEEP_ALIVE}")
    print(f"Timeout: {OLLAMA_TIMEOUT_SECONDS}s")
    print(f"Prompt characters: {len(BENCHMARK_PROMPT)}")
    print("=" * 60)


def main() -> None:
    print_configuration()

    print("\nWarming up Ollama...")

    try:
        call_ollama()
    except Exception as exc:
        print(f"Warm-up failed: {exc}")

    print("\nRunning benchmark...\n")

    for label in ("cold", "warm"):
        start = time.perf_counter()

        data = call_ollama()

        elapsed = time.perf_counter() - start

        format_result(
            label,
            data,
            elapsed,
        )


if __name__ == "__main__":
    main()