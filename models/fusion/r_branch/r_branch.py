import torch


class RBranch:
    """
    Rule/statistics-based R branch.
    Robust to missing values (None).
    """

    def __init__(self,
                 num_classes: int = 3, all_names: list = ["healthy", "stressed", "diseased"],
                 ):
        self.K = num_classes
        self.ALL_NAMES = all_names

    def __call__(self, R_explicit: dict):
        """
        Return soft support based on R statistics.
        If statistics are missing, return neutral evidence.
        """

        support = {label: 0.0 for label in self.ALL_NAMES}

        if R_explicit is None:
            return self._pack(support, confidence=0.0)

        angle = R_explicit.get("angle", None)

        organ = R_explicit.get("organ_ratio", {})
        lesion = R_explicit.get("lesion_stats", {})
        morph = R_explicit.get("morphological_stats", {})

        hp_ratio = organ.get("hp", None)
        stomach_ratio = organ.get("stomach", None)

        red_tail = lesion.get("red_tail_ratio", None)
        speckle = lesion.get("speckle_ratio", None)
        opacity = lesion.get("opacity_ratio", None)

        white_spot = lesion.get("white_spot_match", None)
        black_spot = lesion.get("black_spot_match", None)

        curve = morph.get("curvature_rate", None)
        bend = morph.get("bend_degree", None)

        if self.K == 3:
            if angle is not None and angle == "unknown":
                support["diseased"] += 0.3
                support["stressed"] += 0.3

            # -------------------------
            # organ-based soft rules
            # -------------------------
            if hp_ratio is not None and angle == "dorsal":
                if hp_ratio < 0.3:
                    support["diseased"] += 0.2
                if hp_ratio > 0.5:
                    support["stressed"] += 0.2
            if hp_ratio is not None and angle == "lateral":
                if hp_ratio < 0.25:
                    support["diseased"] += 0.2
                if hp_ratio > 0.65:
                    support["stressed"] += 0.2
            if hp_ratio is not None and angle == "oblique":
                if hp_ratio < 0.35:
                    support["diseased"] += 0.2
                if hp_ratio > 0.6:
                    support["stressed"] += 0.2

            if stomach_ratio is not None:
                if stomach_ratio <= 0.2:
                    support["stressed"] += 0.4

            # -------------------------
            # lesion-based soft rules
            # -------------------------
            if white_spot is not None:
                if white_spot > 0:
                    support["diseased"] += 0.8
            if black_spot is not None:
                if black_spot > 0:
                    support["diseased"] += 0.8

            if red_tail is not None:
                if (red_tail > 0.10) and (red_tail < 0.3):
                    support["healthy"] += 0.4
                if (red_tail > 0.10) and (red_tail < 0.5):
                    support["stressed"] += 0.4
            if red_tail is not None and angle == "dorsal":
                if (red_tail > 0.90) and (red_tail < 1.0):
                    support["diseased"] += 0.4
            if red_tail is not None and angle == "lateral":
                if (red_tail > 0.70) and (red_tail < 0.75):
                    support["diseased"] += 0.4
            if red_tail is not None and angle == "oblique":
                if red_tail > 0.70:
                    support["diseased"] += 0.4

            if speckle is not None:
                if (speckle > 0) and (speckle < 0.3):
                    support["stressed"] += 0.3
                else:
                    support["diseased"] += 0.3

            if opacity is not None:
                if opacity >= 0.50:
                    support["diseased"] += 0.3
                else:
                    support["stressed"] += 0.3

            # -------------------------
            # morphology-based soft rules
            # -------------------------
            if curve is not None and bend is not None:
                if angle == "dorsal" and bend < 0.99:
                    support["stressed"] += 0.3
                    support["diseased"] += 0.3
                elif angle == "lateral" and curve > -0.95:
                    support["stressed"] += 0.3
                    support["diseased"] += 0.3
                elif angle == "oblique":
                    if bend < 0.5:
                        support["healthy"] += 0.3
                    if (bend > 0.5) and (bend < 0.8):
                        support["stressed"] += 0.3
                else:
                    pass

        if self.K == 4:
            # -------------------------
            # lesion-based soft rules
            # -------------------------
            if white_spot is not None:
                if white_spot > 0:
                    support["wssv"] += 0.5
                    support["wssv_bg"] += 0.5

        # -------------------------
        # normalize (optional)
        # -------------------------
        total = sum(support.values())
        if total > 0:
            for k in support:
                support[k] /= total

        confidence = min(total, 1.0)

        return self._pack(support, confidence)

    # -------------------------------------------------

    def _pack(self, support_dict, confidence):

        if len(support_dict.keys()) == 3:
            return {
                "support": {
                    "healthy": torch.tensor(support_dict["healthy"], dtype=torch.float32),
                    "stressed": torch.tensor(support_dict["stressed"], dtype=torch.float32),
                    "diseased": torch.tensor(support_dict["diseased"], dtype=torch.float32),
                },
                "confidence": torch.tensor(confidence, dtype=torch.float32),
            }
        if len(support_dict.keys()) == 4:
            return {
                "support": {
                    "healthy": torch.tensor(support_dict["healthy"], dtype=torch.float32),
                    "wssv": torch.tensor(support_dict["wssv"], dtype=torch.float32),
                    "bg": torch.tensor(support_dict["bg"], dtype=torch.float32),
                    "wssv_bg": torch.tensor(support_dict["wssv_bg"], dtype=torch.float32),
                },
                "confidence": torch.tensor(confidence, dtype=torch.float32),
            }
