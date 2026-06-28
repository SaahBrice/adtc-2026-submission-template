"""Tests for docaware.render (Markdown/DOCX + formula images).

Skips backends that are not installed so the suite stays green on minimal envs.
"""

import importlib.util

import pytest

from docaware.render import render_document

MD = """# Title

Some intro paragraph.

- bullet one
- bullet two

$$ \\frac{a}{b} $$
"""


def test_markdown_render_always_works(tmp_path):
    out = render_document(MD, tmp_path / "doc", fmt="md")
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("# Title")


def test_unsupported_format_raises(tmp_path):
    with pytest.raises(ValueError):
        render_document(MD, tmp_path / "doc", fmt="rtf")


@pytest.mark.skipif(
    importlib.util.find_spec("docx") is None or importlib.util.find_spec("matplotlib") is None,
    reason="python-docx and matplotlib required",
)
def test_docx_render_with_formula(tmp_path):
    out = render_document(MD, tmp_path / "doc", fmt="docx")
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="matplotlib required")
def test_formula_image(tmp_path):
    from docaware.render.formula_img import render_latex_png

    out = render_latex_png(r"\frac{a}{b}", tmp_path / "f.png")
    assert out.exists()
    assert out.stat().st_size > 0
