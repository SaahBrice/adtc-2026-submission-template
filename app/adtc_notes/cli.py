"""adtc_notes/cli.py — Command-line interface (offline, no server needed).

Subcommands:
* ``digitize IMAGE [--format md|docx|pdf]`` — image → formatted document.
* ``add PATH...``                            — index documents for Q&A.
* ``ask "QUESTION"``                          — answer from the indexed corpus.
* ``info``                                    — show config + environment status.

Run as ``python -m adtc_notes <subcommand>`` from the ``app/`` directory.
"""

from __future__ import annotations

import argparse
import sys

from .config import CONFIG
from .errors import ADTCError


def _cmd_digitize(args: argparse.Namespace) -> int:
    from .pipeline import digitize_to_document

    result = digitize_to_document(args.image, fmt=args.format)
    print(f"✅ wrote {result.output_path}")
    for w in result.warnings:
        print(f"⚠️  {w}")
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    from .rag import Retriever

    retriever = Retriever(CONFIG)
    added = retriever.add_documents(args.paths)
    print(f"✅ indexed {added} chunk(s) from {len(args.paths)} file(s)")
    print(f"   index now holds {len(retriever.store)} chunk(s)")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from .rag import Retriever

    retriever = Retriever(CONFIG)
    out = retriever.ask(args.question, top_k=args.top_k)
    print(out["answer"])
    if out["sources"]:
        print("\nSources: " + ", ".join(out["sources"]))
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    import json

    print(json.dumps(CONFIG.as_dict(), indent=2))
    print(f"\nLLM model present:   {CONFIG.llm.model_path.exists()}  ({CONFIG.llm.model_path})")
    print(
        f"Embed model present: {CONFIG.embedding.model_path.exists()}  ({CONFIG.embedding.model_path})"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="adtc_notes", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dig = sub.add_parser("digitize", help="image → formatted document")
    p_dig.add_argument("image", help="path to an image file")
    p_dig.add_argument("--format", choices=("md", "docx", "pdf"), default="md")
    p_dig.set_defaults(func=_cmd_digitize)

    p_add = sub.add_parser("add", help="index documents for Q&A")
    p_add.add_argument("paths", nargs="+", help="document/image paths")
    p_add.set_defaults(func=_cmd_add)

    p_ask = sub.add_parser("ask", help="answer a question from indexed documents")
    p_ask.add_argument("question", help="natural-language question")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.set_defaults(func=_cmd_ask)

    p_info = sub.add_parser("info", help="show config and model status")
    p_info.set_defaults(func=_cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ADTCError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
