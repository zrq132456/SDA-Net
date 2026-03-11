import torch
from typing import Dict, Any

from .r_branch import RBranch


class RProcess:
    """Rule/statistics processor using existing RBranch rules."""

    def __init__(self,
                 num_classes: int = 3, all_names: list = ["healthy", "stressed", "diseased"],):
        super().__init__()
        self.engine = RBranch(num_classes=num_classes, all_names=all_names)

    @torch.no_grad()
    def __call__(self, R_explicit: Dict[str, Any]) -> Dict[str, Any]:
        if R_explicit is None:
            return None
        return self.engine(R_explicit)
