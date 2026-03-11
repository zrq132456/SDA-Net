import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


def _support_dict_to_vec(support, device, ALL_NAMES):
    if support is None:
        return None
    if isinstance(support, torch.Tensor):
        return support.to(device)
    return torch.stack([support[k].to(device) for k in ALL_NAMES], dim=0)


class DecisionModuleV7(nn.Module):
    """
    Decision-oriented evidence integration (2.3.4) with learnable mappings for gamma and tau.

    Baseline mapping is initialized from original rules:
      gamma_base = (1-conf) * 1 + conf * support_T
      tau_base   = clamp( 1 / (1 + scale * support_R), [tau_min, tau_max] )

    Learnable part is a residual:
      gamma = clamp(gamma_base + delta_gamma, 0, 1)
      tau   = clamp(tau_base   + delta_tau, tau_min, tau_max)

    This keeps behavior close to original rules at initialization (fast convergence), while enabling learning.
    """

    def __init__(self,
                 num_classes: int = 3, all_names: list = ["healthy", "stressed", "diseased"],
                 tau_min: float = 0.25, tau_max: float = 2.5, tau_scale: float = 1.5,
                 use_T: bool = True, use_R: bool = True):
        super().__init__()
        self.K = num_classes
        self.ALL_NAMES = all_names
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_scale = tau_scale
        self.use_T = use_T
        self.use_R = use_R

        # Residual nets (initialized to ~0)
        self.gamma_res = nn.Linear(self.K + 1, self.K)
        self.tau_res = nn.Linear(self.K + 1, self.K)

        nn.init.zeros_(self.gamma_res.weight)
        nn.init.zeros_(self.gamma_res.bias)
        nn.init.zeros_(self.tau_res.weight)
        nn.init.zeros_(self.tau_res.bias)

    def forward(
        self,
        e: torch.Tensor,
        T_out: Optional[Dict[str, Any]] = None,
        R_out: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        device = e.device
        e = e.clamp(min=0.0)

        # ---------- gamma ----------
        gamma_base = torch.ones(self.K, device=device)
        sup_T = None
        conf_T = torch.tensor(0.0, device=device)

        if T_out is not None:
            sup_T = _support_dict_to_vec(T_out.get("support"), device, self.ALL_NAMES)
            conf_T = T_out.get("confidence", torch.tensor(0.0)).to(device).view(1)
            conf_T = conf_T.clamp(0.0, 1.0)
            if sup_T is not None:
                gamma_base = (1.0 - conf_T) * torch.ones_like(gamma_base) + conf_T * sup_T.clamp(0.0, 1.0)

        # residual
        if sup_T is not None:
            xT = torch.cat([sup_T.clamp(0.0, 1.0), conf_T], dim=0)  # [K+1]
            delta_gamma = torch.tanh(self.gamma_res(xT))            # bounded residual
            gamma = (gamma_base + delta_gamma).clamp(0.0, 1.0)
        else:
            gamma = gamma_base

        # ---------- tau ----------
        tau_base = torch.ones(self.K, device=device)
        sup_R = None
        conf_R = torch.tensor(0.0, device=device)

        if R_out is not None:
            sup_R = _support_dict_to_vec(R_out.get("support"), device, self.ALL_NAMES)
            conf_R = R_out.get("confidence", torch.tensor(0.0)).to(device).view(1)
            conf_R = conf_R.clamp(0.0, 1.0)
            if sup_R is not None:
                raw = 1.0 / (1.0 + self.tau_scale * sup_R.clamp(0.0, 1.0))
                tau_base = (1.0 - conf_R) * torch.ones_like(tau_base) + conf_R * raw

        if sup_R is not None:
            xR = torch.cat([sup_R.clamp(0.0, 1.0), conf_R], dim=0)  # [K+1]
            delta_tau = torch.tanh(self.tau_res(xR))
            tau = (tau_base + 0.5 * delta_tau).clamp(self.tau_min, self.tau_max)
        else:
            tau = tau_base.clamp(self.tau_min, self.tau_max)

        if not self.use_T:
            gamma = torch.ones_like(gamma)
        if not self.use_R:
            tau = torch.ones_like(tau)

        # ---------- decision mapping (2.3.4) ----------
        e_tilde = gamma * e
        e_hat = e_tilde / tau
        alpha = e_hat + torch.ones_like(e_hat)
        P = alpha / alpha.sum().clamp(min=1e-9)

        return {
            "gamma": gamma,
            "tau": tau,
            "e_tilde": e_tilde,
            "e_hat": e_hat,
            "alpha": alpha,
            "P": P,
        }
