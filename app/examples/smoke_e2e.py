"""examples/smoke_e2e.py — End-to-end smoke test with the real local models.

Exercises the full RAG path (ingest → embed → store → retrieve → LLM answer)
against the sample document, then prints timing. Requires the GGUF models to be
downloaded and llama-cpp-python installed. Run from the ``app/`` directory:

    python examples/smoke_e2e.py
"""

from __future__ import annotations

import time
from pathlib import Path

from adtc_notes.config import CONFIG
from adtc_notes.rag import Retriever

SAMPLE = Path(__file__).parent / "sample_report.md"


def main() -> None:
    print("Models:")
    print(f"  chat : {CONFIG.llm.model_path.exists()}  {CONFIG.llm.model_path.name}")
    print(f"  embed: {CONFIG.embedding.model_path.exists()}  {CONFIG.embedding.model_path.name}")

    r = Retriever(CONFIG)
    t0 = time.time()
    added = r.add_documents([SAMPLE])
    print(f"\nIndexed {added} chunk(s) in {time.time() - t0:.1f}s")

    for q in [
        "What were the Q3 action items and who owns them?",
        "What risks were raised for Q4?",
    ]:
        t0 = time.time()
        out = r.ask(q)
        print(f"\nQ: {q}")
        print(f"A ({time.time() - t0:.1f}s): {out['answer']}")
        print(f"Sources: {', '.join(out['sources'])}")


if __name__ == "__main__":
    main()
