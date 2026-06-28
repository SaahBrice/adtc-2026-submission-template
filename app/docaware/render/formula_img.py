"""docaware/render/formula_img.py — Render LaTeX math to PNG without a TeX install.

Matplotlib's built-in *mathtext* engine renders a large LaTeX subset to images
with no system TeX dependency — much lighter than TeX Live on the target laptop.
Used to embed equations into DOCX/PDF where raw LaTeX cannot be typeset directly.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import BackendNotInstalledError


def render_latex_png(latex: str, out_path: str | Path, *, fontsize: int = 16) -> Path:
    """Render a LaTeX math string to a transparent PNG.

    Args:
        latex: Math content WITHOUT surrounding ``$`` (e.g. ``r"\\frac{a}{b}"``).
        out_path: Destination .png path.
        fontsize: Point size for the rendered equation.

    Returns:
        The output path.

    Raises:
        BackendNotInstalledError: If matplotlib is not installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed, fully offline
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise BackendNotInstalledError("matplotlib not installed: pip install matplotlib") from exc

    out_path = Path(out_path)
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=fontsize)
    fig.savefig(
        out_path,
        dpi=200,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    return out_path
