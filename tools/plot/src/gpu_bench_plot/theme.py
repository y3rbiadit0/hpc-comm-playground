"""Palette, chart chrome, and the backend-to-colour mapping."""

from __future__ import annotations

from ._mpl import plt

# Backends in a fixed order so a colour always means the same backend, whatever
# subset a given results tree happens to contain. Colour follows the entity, not
# its position in the current figure.
BACKEND_ORDER = (
    "cuda_mpi",
    "cuda_nccl",
    "cuda_nvshmem",
    "oshmpi",
    "sycl_mpi",
    "sycl_oneccl",
    "sycl_oneccl_oshmpi",
)
BASELINE_BACKEND = "cuda_mpi"

# The phases of a cg_step iteration, in execution order. These need their own
# four colours: reusing the backend slots would make one hue mean a backend in
# one figure and a phase in another.
PHASE_ORDER = ("pack", "halo", "compute", "reduce")

# Categorical slots 1-7, validated for adjacent-pair colour-vision-deficiency
# separation against both surfaces (worst adjacent CVD dE 9.1 light / 8.4 dark,
# normal-vision 19.6 / 19.3). The ORDER is the safety mechanism, not decoration:
# reordering these invalidates the check. Re-run the validator if you change them.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e2e1dd",
        "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"),
        # Diverging poles for the speedup scale, low -> high, neutral in the middle.
        "diverging": ("#2a78d6", "#f0efec", "#e34948"),
        # Phase slots, in PHASE_ORDER. Drawn from the same validated categorical
        # ramp, reordered so adjacent phases stay separable: worst adjacent CVD
        # dE 9.2 light / 9.4 dark against these surfaces.
        "phases": ("#eda100", "#2a78d6", "#1baf7a", "#eb6834"),
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#383835",
        "series": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"),
        "diverging": ("#3987e5", "#383835", "#e66767"),
        "phases": ("#c98500", "#3987e5", "#199e70", "#d95926"),
    },
}


def apply_theme(theme: dict) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": theme["surface"],
            "axes.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "text.color": theme["text"],
            "axes.labelcolor": theme["muted"],
            "axes.titlecolor": theme["text"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            # Recessive chrome: hairline, solid, one shade off the surface.
            "axes.edgecolor": theme["grid"],
            "axes.linewidth": 0.8,
            "grid.color": theme["grid"],
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "font.size": 9,
            "axes.titlesize": 10,
            "legend.frameon": False,
            "figure.dpi": 140,
        }
    )


def colour_for(backend: str, theme: dict) -> str:
    """The fixed slot for this backend, or muted grey for one we do not know."""
    series = theme["series"]
    try:
        return series[BACKEND_ORDER.index(backend)]
    except (ValueError, IndexError):
        return theme["muted"]


def phase_colour(phase: str, theme: dict) -> str:
    """The fixed slot for this phase, or muted grey for one we do not know."""
    try:
        return theme["phases"][PHASE_ORDER.index(phase)]
    except (ValueError, IndexError, KeyError):
        return theme["muted"]


def style_axes(ax, theme: dict) -> None:
    ax.grid(True, which="major", alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure_legend(fig, handles, labels, theme: dict) -> None:
    # "outside lower center" makes constrained_layout allocate a strip for the
    # legend instead of drawing it on top of the bottom row of panels.
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(len(labels), 4),
        labelcolor=theme["muted"],
    )
