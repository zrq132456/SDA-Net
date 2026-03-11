import torch


class TBranch_V3:
    """
    TBranch v3
    - Hierarchical rule-based inference
    - Robust to missing fields
    - Designed for veto / correction (NOT for training)
    """

    def __init__(self,
                 num_classes: int = 3, all_names: list = ["healthy", "stressed", "diseased"],
                 ):
        self.K = num_classes
        self.ALL_NAMES = all_names

    # -------------------------------------------------
    def __call__(self, T_explicit: dict):
        """
        Input:
            T_explicit: dict (may be sparse)

        Output:
            {
              "support": {healthy, stressed, diseased},
              "confidence": float,
              "rules": list[str]
            }
        """

        support = {label: 0.0 for label in self.ALL_NAMES}
        fired_rules = []

        if T_explicit is None:
            return self._pack(support, fired_rules)
        if self.K == 3:
            # =================================================
            # 1. Disease-dominant evidence (HIGH PRIORITY)
            # =================================================
            spot = T_explicit.get("Spot", {})
            resp = T_explicit.get("Respiratory", {})
            surface = T_explicit.get("Surface", {})
            digest = T_explicit.get("Digest", {})

            # ---- white spot ----
            if spot.get("white_spot") == "true":
                support["diseased"] += 0.9
                fired_rules.append("white_spot → diseased, suspect: WSSV")
            # ---- black spot ----
            if spot.get("black_spot") == "true":
                support["diseased"] += 0.9
                fired_rules.append("black_spot → diseased, suspect: Fungi / Bacteria")
            # ----- speckle ------
            if spot.get("speckle") in ["moderate", "severe"]:
                support["diseased"] += 0.6
                fired_rules.append("speckle → diseased, suspect: Bacteria / Fungi")

            # ---- black gill / dirty gill ----
            if resp.get("gill_color") == "black":
                support["diseased"] += 0.6
                fired_rules.append("black_gill → diseased")

            if resp.get("gill_dirty") == "true":
                support["stressed"] += 0.3
                fired_rules.append("dirty_gill → stressed")

            # ---- stomach atrophy ----
            if digest.get("stomach_atrophy") == "true" and digest.get("get_empty") == "true":
                support["stressed"] += 0.2
                fired_rules.append("stomach_atrophy → stressed: food shortage (confirmed)")
            # ---- hepatopancreas atrophy ----
            if digest.get("hp_shape") == "atrophy":
                support["diseased"] += 0.3
                fired_rules.append("hp_atrophy → diseased")
            if digest.get("hp_color") == "pale":
                support["diseased"] += 0.1
                fired_rules.append("hp_abnormal → diseased")
            if digest.get("hp_color") == "red":
                support["stressed"] += 0.1
                fired_rules.append("hp_abnormal → stressed: High-density food shortage (suspected)")
                support["diseased"] += 0.1
                fired_rules.append("hp_abnormal → diseased: Enteritis (suspected)")

            # ---- tail redness (severe) ----
            if surface.get("tail_red") == "severe":
                support["diseased"] += 0.4
                fired_rules.append("severe_tail_red → diseased")
            # ---- tail redness (moderate) ----
            if surface.get("tail_red") == "moderate":
                support["diseased"] += 0.2
                fired_rules.append("moderate_tail_red → diseased (suspected)")

            # =================================================
            # 2. Stress evidence (SECOND PRIORITY)
            # =================================================
            trauma = T_explicit.get("Trauma", {})
            body = T_explicit.get("Body", {})

            # ---- shell_crack + white_spot (severe) ----
            if trauma.get("shell_crack") == "true" and spot.get("white_spot") == "true":
                support["diseased"] += 0.8
                fired_rules.append("shell_crack + white_spot → diseased (WSSV confirmed)")

            # ---- tail redness (edge) ----
            if surface.get("tail_red") == "edge":
                support["stressed"] += 0.4
                fired_rules.append("edge_tail_red → stressed")
            # ---- tail redness (mild) ----
            if surface.get("tail_red") == "mild":
                support["stressed"] += 0.2
                fired_rules.append("mild_tail_red → stressed (suspected)")

            # ---- Trauma ----
            if trauma.get("eye_loss") == "true":
                support["stressed"] += 0.2
                fired_rules.append("eye_loss → stressed")
            if trauma.get("antenna_loss") == "true":
                support["stressed"] += 0.2
                fired_rules.append("antenna_loss → stressed")
            if trauma.get("limb_loss") == "true":
                support["stressed"] += 0.2
                fired_rules.append("limb_loss → stressed")
            if trauma.get("shell_crack") == "true":
                support["stressed"] += 0.2
                fired_rules.append("shell_crack → stressed")

            # ---- Surface: limb ----
            if surface.get("limb_red") in ["tip", "root", "entire"]:
                support["stressed"] += 0.2
                fired_rules.append("limb_red → stressed")
            if surface.get("limb_black") in ["tip", "root", "entire"]:
                support["stressed"] += 0.1
                fired_rules.append("limb_black → stressed")
                support["diseased"] += 0.1
                fired_rules.append("limb_black → diseased")

            if body.get("opacity") in ["moderate", "severe"]:
                support["diseased"] += 0.3
                fired_rules.append("opacity → diseased: Muscle necrosis (suspected)")
            if body.get("opacity") == "mild":
                support["stressed"] += 0.1
                fired_rules.append("opacity → stressed")

            if body.get("shell_color") in ["uneven", "red"]:
                support["diseased"] += 0.2
                fired_rules.append("abnormal_shell_color → diseased")

            # ---- morphology ----
            if body.get("morphology") == "rigor":
                support["stressed"] += 0.3
                fired_rules.append("rigor_morphology → stressed")
            if body.get("morphology") == "deform":
                support["diseased"] += 0.3
                fired_rules.append("deform_morphology → diseased")

            # =================================================
            # 3. Healthy evidence (ONLY if no abnormality)
            # =================================================
            visible = T_explicit.get("Visible_flags", {})
            if (
                    visible.get("body") == "complete"
                    and spot.get("white_spot") == "false"
                    and spot.get("black_spot") == "false"
                    and surface.get("tail_red") == "none"
                    and resp.get("gill_dirty") == "false"
                    and resp.get("hp_shape") == "normal"
                    and body.get("opacity") == "none"
                    and body.get("morphology") != "deform"
                    and all(v == "false" for v in trauma.values())
            ):
                support["healthy"] += 0.6
                fired_rules.append("no_visible_abnormality → healthy")

            # =================================================
            # 4. Normalize & confidence
            # =================================================
            total = sum(support.values())
            if total > 0:
                for k in support:
                    support[k] /= total

            confidence = min(total, 1.0)

            return self._pack(support, fired_rules, confidence)

        if self.K == 4:
            # =================================================
            # 1. Disease-dominant evidence (HIGH PRIORITY)
            # =================================================
            spot = T_explicit.get("Spot", {})
            resp = T_explicit.get("Respiratory", {})

            if spot.get("white_spot") == "true" and resp.get("gill_color") == "black":
                support["wssv_bg"] = 1.0
                fired_rules.append("white_spot + black_gill → suspect: wssv_bg")
                confidence = 1.0
                return self._pack(support, fired_rules, confidence)

            if spot.get("white_spot") != "true" and resp.get("gill_color") != "black":
                support["healthy"] = 1.0
                confidence = 0.9
                fired_rules.append("no lesion found -> please provide more details")
                return self._pack(support, fired_rules, confidence)

            if spot.get("white_spot") == "true":
                support["wssv"] = 1.0
                confidence = 0.9
                fired_rules.append("white_spot → suspect: WSSV")
                return self._pack(support, fired_rules, confidence)

            if resp.get("gill_color") == "black":
                support["bg"] = 1.0
                confidence = 0.9
                fired_rules.append("black_gill → suspect: bg")
                return self._pack(support, fired_rules, confidence)

            return self._pack(support, fired_rules, confidence=0.0)

    # -------------------------------------------------
    def _pack(self, support, rules, confidence=None):
        if confidence is None:
            confidence = sum(support.values())

        if len(support.keys()) == 3:
            return {
                "support": {
                    "healthy": torch.tensor(support["healthy"], dtype=torch.float32),
                    "stressed": torch.tensor(support["stressed"], dtype=torch.float32),
                    "diseased": torch.tensor(support["diseased"], dtype=torch.float32),
                },
                "confidence": torch.tensor(confidence, dtype=torch.float32),
                "rules": rules,
            }
        if len(support.keys()) == 4:
            return {
                "support": {
                    "healthy": torch.tensor(support["healthy"], dtype=torch.float32),
                    "wssv": torch.tensor(support["wssv"], dtype=torch.float32),
                    "bg": torch.tensor(support["bg"], dtype=torch.float32),
                    "wssv_bg": torch.tensor(support["wssv_bg"], dtype=torch.float32),
                },
                "confidence": torch.tensor(confidence, dtype=torch.float32),
                "rules": rules,
            }