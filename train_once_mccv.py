import os
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.dataloader import ShrimpDataset
from dataset.utils import set_global_seed, load_all_ids, split_ids

from models.backbone.backbone_multiscale import VisualEvidenceNet
from models.fusion.decision_module import DecisionModuleV7
from models.fusion.t_branch.t_process import TProcess
from models.fusion.r_branch.r_process import RProcess

from sklearn.metrics import accuracy_score, f1_score, classification_report

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_symlink_or_copy(src: str, dst: str):
    """
    Create symlink if possible; fallback to copy.
    """
    try:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    except Exception:
        shutil.copy(src, dst)


def json_dump(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def jsonl_append(path: str, record: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


@dataclass
class EarlyStopping:
    monitor: str = "macro_f1"
    mode: str = "max"         # "max" or "min"
    patience: int = 5
    min_delta: float = 1e-3
    min_epoch: int = 0

    best_score: Optional[float] = None
    best_epoch: int = -1
    wait: int = 0

    def step(self, epoch: int, metrics: Dict[str, Any]) -> Tuple[bool, bool]:
        """
        Returns:
          stop (bool): whether to early stop
          is_best (bool): whether current epoch is best
        """
        score = metrics.get(self.monitor, None)
        if score is None:
            return False, False

        # ===== warm-up: no early stop before min_epoch =====
        if epoch < self.min_epoch:
            # still track best, but never stop
            if self.best_score is None or (
                    (self.mode == "max" and score > self.best_score)
                    or (self.mode == "min" and score < self.best_score)
            ):
                self.best_score = score
                self.best_epoch = epoch
                return False, True
            return False, False

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.wait = 0
            return False, True

        improved = False
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.wait = 0
            return False, True

        self.wait += 1
        if self.wait >= self.patience:
            return True, False
        return False, False


def make_run_dir(work_dir: str, note: str, seed: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_note = "".join([c if c.isalnum() or c in "-_+" else "_" for c in (note or "exp")])
    run_name = f"{ts}_{safe_note}_seed{seed}"
    run_dir = os.path.join(work_dir, run_name)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "checkpoints"))
    return run_dir


def extract_support_conf_rules(out: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize T_out / R_out fields into JSON-serializable dict.
    """
    if out is None:
        return {"present": False}

    rec = {"present": True}

    conf = out.get("confidence", None)
    if isinstance(conf, torch.Tensor):
        conf = float(conf.detach().cpu().item())
    rec["confidence"] = conf

    support = out.get("support", None)
    if isinstance(support, dict):
        rec["support"] = {k: float(v.detach().cpu().item()) if isinstance(v, torch.Tensor) else float(v)
                          for k, v in support.items()}
    elif isinstance(support, torch.Tensor):
        rec["support"] = [float(x) for x in support.detach().cpu().tolist()]
    else:
        rec["support"] = support

    rules = out.get("rules", None)
    if rules is not None:
        rec["rules"] = rules
    return rec


def train_once(args) -> Dict[str, Any]:
    """
    One experiment = one seed split = one run_dir.
    Returns cls_metrics (dict).
    """

    # -------------------------
    # setup
    # -------------------------
    set_global_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = make_run_dir(args.work_dir, args.note, args.seed)

    paths = {
        "params": os.path.join(run_dir, "params.json"),
        "train_loss": os.path.join(run_dir, "train_loss.json"),
        "epoch_summary": os.path.join(run_dir, "epoch_summary.json"),
        "train_sample": os.path.join(run_dir, "train_sample.jsonl"),
        "eval_sample": os.path.join(run_dir, "eval_sample.jsonl"),
        "cls_metrics": os.path.join(run_dir, "cls_metrics.json"),
        "meta": os.path.join(run_dir, "meta.json"),
        "ckpt_dir": os.path.join(run_dir, "checkpoints"),
    }

    LABEL_MAP = args.label_map
    INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
    K = args.num_classes
    ALL_LABELS = list(range(K))
    ALL_NAMES = [INV_LABEL_MAP[i] for i in ALL_LABELS]

    # write params.json
    params = {
        "note": args.note,
        "seed": args.seed,
        "dataset_root": args.dataset_root,
        "num_classes": K,
        "label_map": args.label_map,
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_conf": args.lambda_conf,
        "lambda_risk": args.lambda_risk,
        "train_ratio": args.train_ratio,
        "use_T": args.use_T,
        "use_R": args.use_R,
        "use_dirichlet": args.use_dirichlet,
        "label_map": LABEL_MAP,
        "vision_model": args.backbone,
        "early_stopping": {
            "monitor": args.es_monitor,
            "mode": args.es_mode,
            "patience": args.es_patience,
            "min_delta": args.es_min_delta,
            "min_epoch": args.es_min_epoch,
        }
    }
    json_dump(paths["params"], params)

    # reset loss/epoch files
    json_dump(paths["train_loss"], [])
    json_dump(paths["epoch_summary"], [])

    # -------------------------
    # ids split (val fold early stopping)
    # -------------------------
    all_ids = load_all_ids(dataset_root=args.dataset_root, image_dir="images")
    train_ids, val_ids = split_ids(all_ids, train_ratio=args.train_ratio, seed=args.seed)

    train_dataset = ShrimpDataset(dataset_root=args.dataset_root, ids=train_ids, load_image=True)
    val_dataset = ShrimpDataset(dataset_root=args.dataset_root, ids=val_ids, load_image=True)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # -------------------------
    # models
    # -------------------------
    vision = VisualEvidenceNet(num_classes=K, pretrained=True, backbone=args.backbone).to(device)
    decision = DecisionModuleV7(num_classes=K, all_names=ALL_NAMES,
                                use_T=args.use_T, use_R=args.use_R).to(device)

    T_proc = TProcess(num_classes=K, all_names=ALL_NAMES)
    R_proc = RProcess(num_classes=K, all_names=ALL_NAMES)

    # optimizer includes BOTH vision + decision (mapping/regulation learnable)
    optimizer = torch.optim.Adam(list(vision.parameters()) + list(decision.parameters()), lr=args.lr)

    # early stopping
    es = EarlyStopping(
        monitor=args.es_monitor,
        mode=args.es_mode,
        patience=args.es_patience,
        min_delta=args.es_min_delta,
        min_epoch=args.es_min_epoch,
    )

    train_loss_records = []
    epoch_summaries = []

    # -------------------------
    # training loop
    # -------------------------
    global_step = 0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        vision.train()
        decision.train()

        for batch in train_loader:
            global_step += 1

            sample_id = batch.get("id", None)
            if isinstance(sample_id, list):
                sample_id = sample_id[0]

            label_str = batch.get("label", None)
            if label_str not in LABEL_MAP:
                continue
            y = torch.tensor(LABEL_MAP[label_str], dtype=torch.long, device=device)

            if batch.get("I_explicit") is None:
                continue

            # ---- forward: visual evidence
            I_out = vision(batch["I_explicit"])
            e = I_out["e"]  # [K]

            # ---- rule outputs (not learned): support/conf/rules
            T_out = T_proc(batch.get("T_explicit")) if (args.use_T and batch.get("T_explicit") is not None) else None
            R_out = R_proc(batch.get("R_explicit")) if (args.use_R and batch.get("R_explicit") is not None) else None

            out = decision(e=e, T_out=T_out, R_out=R_out)  # alpha/gamma/tau/P

            # ---- losses
            alpha = out["alpha"].clamp(min=1e-8)
            alpha0 = alpha.sum().clamp(min=1e-8)

            if args.use_dirichlet:
                # decision supervision on Dirichlet concentration
                L_cls = -torch.log(alpha[y] / alpha0)
            else:
                # softmax control (debug): pseudo-logits from evidence
                logits = torch.log(e + 1e-6)
                L_cls = nn.CrossEntropyLoss()(logits.unsqueeze(0), y.unsqueeze(0))

            gamma = out["gamma"].clamp(0.0, 1.0)
            tau = out["tau"].clamp(min=1e-6)

            L_conf = torch.sum(torch.abs((1.0 - gamma) * e))
            L_risk = torch.sum((tau - tau.mean()) ** 2)

            loss = L_cls + args.lambda_conf * L_conf + args.lambda_risk * L_risk

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- record train_loss.json
            rec_loss = {
                "epoch": epoch,
                "step": global_step,
                "L_cls": float(L_cls.detach().cpu().item()),
                "L_conf": float(L_conf.detach().cpu().item()),
                "L_risk": float(L_risk.detach().cpu().item()),
                "total": float(loss.detach().cpu().item()),
            }
            train_loss_records.append(rec_loss)

            # ---- record train_sample.jsonl (per sample decision trace)
            pred = int(out["P"].argmax().item())
            sample_rec = {
                "epoch": epoch,
                "step": global_step,
                "sample_id": sample_id,
                "label": label_str,
                "label_id": int(y.item()),
                "prediction": pred,
                "prediction_name": INV_LABEL_MAP.get(pred, str(pred)),
                "e": [float(x) for x in e.detach().cpu().tolist()],
                "alpha": [float(x) for x in alpha.detach().cpu().tolist()],
                "gamma": [float(x) for x in gamma.detach().cpu().tolist()],
                "tau": [float(x) for x in tau.detach().cpu().tolist()],
                "P": [float(x) for x in out["P"].detach().cpu().tolist()],
                "beta_scales": [float(x) for x in I_out["debug"]["beta_scales"].detach().cpu().tolist()],
                "T": extract_support_conf_rules(T_out),
                "R": extract_support_conf_rules(R_out),
            }
            jsonl_append(paths["train_sample"], sample_rec)

            # ---- periodic flush train_loss.json
            if global_step % args.flush_every == 0:
                json_dump(paths["train_loss"], train_loss_records)

        # end epoch -> flush loss
        json_dump(paths["train_loss"], train_loss_records)

        # -------------------------
        # eval on val fold (for early stopping)
        # -------------------------
        vision.eval()
        decision.eval()

        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in val_loader:
                sample_id = batch.get("id", None)
                if isinstance(sample_id, list):
                    sample_id = sample_id[0]
                label_str = batch.get("label", None)
                if label_str not in LABEL_MAP:
                    continue
                y_i = LABEL_MAP[label_str]

                if batch.get("I_explicit") is None:
                    continue

                I_out = vision(batch["I_explicit"])
                e = I_out["e"]

                T_out = T_proc(batch.get("T_explicit")) if (args.use_T and batch.get("T_explicit") is not None) else None
                R_out = R_proc(batch.get("R_explicit")) if (args.use_R and batch.get("R_explicit") is not None) else None

                out = decision(e=e, T_out=T_out, R_out=R_out)
                pred = int(out["P"].argmax().item())

                y_true.append(y_i)
                y_pred.append(pred)

                # record eval sample trace
                alpha = out["alpha"].clamp(min=1e-8)
                gamma = out["gamma"].clamp(0.0, 1.0)
                tau = out["tau"].clamp(min=1e-6)

                eval_rec = {
                    "epoch": epoch,
                    "sample_id": sample_id,
                    "label": label_str,
                    "label_id": int(y_i),
                    "prediction": pred,
                    "prediction_name": INV_LABEL_MAP.get(pred, str(pred)),
                    "e": [float(x) for x in e.detach().cpu().tolist()],
                    "alpha": [float(x) for x in alpha.detach().cpu().tolist()],
                    "gamma": [float(x) for x in gamma.detach().cpu().tolist()],
                    "tau": [float(x) for x in tau.detach().cpu().tolist()],
                    "P": [float(x) for x in out["P"].detach().cpu().tolist()],
                    "beta_scales": [float(x) for x in I_out["debug"]["beta_scales"].detach().cpu().tolist()],
                    "T": extract_support_conf_rules(T_out),
                    "R": extract_support_conf_rules(R_out),
                }
                jsonl_append(paths["eval_sample"], eval_rec)

        #
        present_labels = sorted(set(y_true))
        present_label_names = [INV_LABEL_MAP[i] for i in present_labels]

        metrics = compute_cls_metrics(y_true, y_pred,
                                      all_labels=ALL_LABELS,
                                      all_names=ALL_NAMES,
                                      present_label_names=present_label_names,
                                      )

        # early stopping check
        stop, is_best = es.step(epoch, metrics)

        # epoch summary record
        epoch_summary = {
            "epoch": epoch,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "healthy_precision": metrics["healthy_precision"],
            "stressed_recall": metrics["stressed_recall"],
            "diseased_recall": metrics["diseased_recall"],
            "weighted_f1": metrics["weighted_f1"],
            "is_best": is_best,
        }
        epoch_summaries.append(epoch_summary)
        json_dump(paths["epoch_summary"], epoch_summaries)

        # save checkpoints
        ckpt_dir = paths["ckpt_dir"]
        v_epoch = os.path.join(ckpt_dir, f"visual_epoch_{epoch:03d}.pth")
        d_epoch = os.path.join(ckpt_dir, f"decision_epoch_{epoch:03d}.pth")

        torch.save(vision.state_dict(), v_epoch)
        torch.save(decision.state_dict(), d_epoch)

        safe_symlink_or_copy(v_epoch, os.path.join(ckpt_dir, "visual_last.pth"))
        safe_symlink_or_copy(d_epoch, os.path.join(ckpt_dir, "decision_last.pth"))

        if is_best:
            best_epoch = epoch
            safe_symlink_or_copy(v_epoch, os.path.join(ckpt_dir, "visual_best.pth"))
            safe_symlink_or_copy(d_epoch, os.path.join(ckpt_dir, "decision_best.pth"))

        # meta update each epoch (useful if training interrupted)
        meta = {
            "run_dir": run_dir,
            "best_epoch": best_epoch,
            "best_score": es.best_score,
            "stopped_epoch": epoch if stop else None,
        }
        json_dump(paths["meta"], meta)

        # console short print (still OK)
        print(f"[{run_dir}] epoch={epoch} acc={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f} "
              f"{'(BEST)' if is_best else ''} wait={es.wait}/{es.patience}")

        if stop:
            print(f"[EarlyStop] stop at epoch={epoch}, best_epoch={es.best_epoch}, best_{es.monitor}={es.best_score}")
            break

    # final metrics saved (use BEST epoch metrics from last val eval for simplicity)
    json_dump(paths["cls_metrics"], metrics)

    return {
        "run_dir": run_dir,
        "metrics": metrics,
        "best_epoch": es.best_epoch,
        "best_score": es.best_score,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="../datasets")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lambda_conf", type=float, default=0.6)
    parser.add_argument("--lambda_risk", type=float, default=0.3)
    parser.add_argument("--train_ratio", type=float, default=0.4)
    parser.add_argument("--work_dir", type=str, default="work_dir")
    parser.add_argument("--note", type=str, default="exp")

    parser.add_argument("--use_T", action="store_true", default=True)
    parser.add_argument("--use_R", action="store_true", default=True)
    parser.add_argument("--use_dirichlet", action="store_true", default=True)

    parser.add_argument("--backbone", type=str, default="resnet50", help="vgg16/inception_v3/mobilenet_v2/xception")

    parser.add_argument("--flush_every", type=int, default=50)

    parser.add_argument("--es_monitor", type=str, default="accuracy")
    parser.add_argument("--es_mode", type=str, default="max")
    parser.add_argument("--es_patience", type=int, default=5)
    parser.add_argument("--es_min_delta", type=float, default=1e-4)
    parser.add_argument("--es_min_epoch", type=int, default=5)

    args = parser.parse_args()

    res = train_once(args)
    print("Done:", res)
