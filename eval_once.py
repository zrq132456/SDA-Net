import os
import json
import argparse
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from dataset.dataloader import ShrimpDataset
from dataset.utils import set_global_seed, load_all_ids, split_ids

from models.backbone.backbone_multiscale import VisualEvidenceNet
from models.fusion.t_branch.t_process import TProcess
from models.fusion.r_branch.r_process import RProcess
from models.fusion.decision_module import DecisionModuleV7


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]

def compute_cls_metrics(y_true: List[int], y_pred: List[int],
                        all_labels: Dict[str, int] = None,
                        all_names: List[str] = None,
                        present_label_names: List = None,
                        ) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true, y_pred,
        labels=all_labels,  # ← 显式固定任务类别
        target_names=all_names,  # ← 顺序严格一致
        output_dict=True,
        zero_division=0
    )

    # -------- macro precision / recall --------
    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]

    # -------- per-class focus metrics --------
    def safe_get(cls_name: str, key: str):
        if cls_name in report:
            return report[cls_name].get(key, 0.0)
        return 0.0

    focus_metrics = {
        "healthy_precision": safe_get("healthy", "precision"),
        "stressed_recall": safe_get("stressed", "recall"),
        "diseased_recall": safe_get("diseased", "recall"),
    }

    per_class_recall = {
        name: report[name]["recall"]
        for name in all_names
        if name in report
    }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        **focus_metrics,
        "per_class_recall": per_class_recall,
        "report": report,
        "n_samples": len(y_true),
        "present_labels": present_label_names,
    }

def eval_once(
    dataset_root: str,
    vision_ckpt: str,
    decision_ckpt: str,
    seed: int = 42,
    use_T: bool = True,
    use_R: bool = True,
    save_dir: Optional[str] = None,
    label_map: Dict[str, int] = None,
    num_classes: int = None,
    backbone: str = "resnet50",
):
    """
    Run evaluation once on the validation split.
    """

    set_global_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    LABEL_MAP = label_map
    INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
    K = num_classes
    ALL_LABELS = list(range(K))
    ALL_NAMES = [INV_LABEL_MAP[i] for i in ALL_LABELS]

    # -------------------------
    # load ids (same split as train)
    # -------------------------
    all_ids = load_all_ids(dataset_root=dataset_root, image_dir="images")
    _, val_ids = split_ids(all_ids, train_ratio=0.8, seed=seed)

    dataset = ShrimpDataset(
        dataset_root=dataset_root,
        ids=val_ids,
        load_image=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # -------------------------
    # models
    # -------------------------
    vision = VisualEvidenceNet(num_classes=K, pretrained=False, backbone=backbone).to(device)
    decision = DecisionModuleV7(
        num_classes=K,
        all_names=ALL_NAMES,
        use_T=use_T,
        use_R=use_R,
    ).to(device)

    vision.load_state_dict(torch.load(vision_ckpt, map_location=device))
    decision.load_state_dict(torch.load(decision_ckpt, map_location=device))

    vision.eval()
    decision.eval()

    T_proc = TProcess(num_classes=K, all_names=ALL_NAMES)
    R_proc = RProcess(num_classes=K, all_names=ALL_NAMES)

    # -------------------------
    # containers
    # -------------------------
    y_true: List[int] = []
    y_pred: List[int] = []
    records: List[Dict[str, Any]] = []

    # -------------------------
    # eval loop
    # -------------------------
    with torch.no_grad():
        for sample in loader:
            sample_id = sample.get("id", None)
            label_str = sample.get("label", None)

            if label_str not in LABEL_MAP:
                continue
            if sample.get("I_explicit") is None:
                continue

            y = LABEL_MAP[label_str]
            y_true.append(y)

            # visual
            I_out = vision(sample["I_explicit"])
            e = I_out["e"]

            # structured / statistical
            T_out = (
                T_proc(sample.get("T_explicit"))
                if (use_T and sample.get("T_explicit") is not None)
                else None
            )
            R_out = (
                R_proc(sample.get("R_explicit"))
                if (use_R and sample.get("R_explicit") is not None)
                else None
            )

            # decision
            out = decision(e=e, T_out=T_out, R_out=R_out)

            P = out["P"]
            pred = int(P.argmax().item())
            y_pred.append(pred)

            # -------- record --------
            record = {
                "id": sample_id,
                "label": label_str,
                "label_idx": y,
                "pred_idx": pred,
                "pred_label": INV_LABEL_MAP[pred],
                "P": P.detach().cpu().tolist(),
                "e": e.detach().cpu().tolist(),
                "alpha": out["alpha"].detach().cpu().tolist()
                if "alpha" in out
                else None,
                "gamma": out["gamma"].detach().cpu().tolist()
                if out.get("gamma") is not None
                else None,
                "tau": out["tau"].detach().cpu().tolist()
                if out.get("tau") is not None
                else None,
            }

            if T_out is not None:
                record["T"] = {
                    "confidence": float(T_out.get("confidence", -1)),
                    "support": {
                        k: float(v) for k, v in T_out.get("support", {}).items()
                    },
                    "rules": T_out.get("rules", []),
                }
            else:
                record["T"] = None

            if R_out is not None:
                record["R"] = {
                    "confidence": float(R_out.get("confidence", -1)),
                    "support": {
                        k: float(v) for k, v in R_out.get("support", {}).items()
                    },
                }
            else:
                record["R"] = None

            records.append(record)

    #
    present_labels = sorted(set(y_true))
    present_label_names = [INV_LABEL_MAP[i] for i in present_labels]

    # -------------------------
    # metrics
    # -------------------------
    metrics = compute_cls_metrics(
        y_true,
        y_pred,
        all_labels=ALL_LABELS,
        all_names=ALL_NAMES,
        present_label_names=present_label_names,
    )

    # -------------------------
    # save
    # -------------------------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "eval_sample.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        with open(os.path.join(save_dir, "cls_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="../datasets")
    parser.add_argument("--vision_ckpt", type=str, required=True)
    parser.add_argument("--decision_ckpt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_T", action="store_true", default=True)
    parser.add_argument("--use_R", action="store_true", default=True)
    parser.add_argument("--use_dirichlet", action="store_true", default=True)
    parser.add_argument("--backbone", type=str, default="resnet50", help="vgg16/inception_v3/mobilenet_v2/xception")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--label_map", type=Dict, default={"healthy": 0, "stressed": 1, "diseased": 2})
    parser.add_argument("--num_classes", type=int, default=3)

    args = parser.parse_args()

    metrics = eval_once(
        dataset_root=args.dataset_root,
        vision_ckpt=args.vision_ckpt,
        decision_ckpt=args.decision_ckpt,
        seed=args.seed,
        use_T=args.use_T,
        use_R=args.use_R,
        backbone=args.backbone,
        use_dirichlet=args.use_dirichlet,
        save_dir=args.save_dir,
        label_map=args.label_map,
        num_classes=args.num_classes,
    )

    print("Accuracy:", metrics["accuracy"])
    print("Macro-F1:", metrics["macro_f1"])
    print("Weighted-F1:", metrics["weighted_f1"])
