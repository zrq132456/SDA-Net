import torch
from typing import Dict, Any

from .t_branch_v3 import TBranch_V3 as TBranch


class TProcess:
    """Rule-based structured indicator processor.
    Uses existing TBranch rules to produce:
      - support: dict[str, Tensor] (3 classes)
      - confidence: Tensor scalar in [0, +inf) (we will clamp to [0,1] in decision module)
      - rules: fired rules
    """

    def __init__(self,
                 num_classes: int = 3, all_names: list = ["healthy", "stressed", "diseased"],):
        super().__init__()
        self.engine = TBranch(num_classes=num_classes, all_names=all_names)

    @torch.no_grad()
    def __call__(self, T_explicit: Dict[str, Any]) -> Dict[str, Any]:
        if T_explicit is None:
            return None
        out = self.engine(T_explicit)
        return out
