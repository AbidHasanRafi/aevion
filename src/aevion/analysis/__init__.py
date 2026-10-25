from . import figures, style
from .style import use_style

__all__ = ["figures", "style", "use_style"]  # diagnostics, figures_diag, templates and appendix are imported where needed (Matplotlib loads lazily)
