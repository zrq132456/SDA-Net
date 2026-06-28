import os
import json
import copy
import argparse
import yaml
from typing import List, Dict, Any
from types import SimpleNamespace

from train_once import train_once
from eval_once import eval_once

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config yaml")
    # optional overwrite
    # ===== basic =====
    parser.add_argument("--dataset_root", type=str, default="../datasets")
    parser.add_argument("--work_dir", type=str, default="work_dir")
    parser.add_argument("--note", type=str, default="cv_exp")
    # ===== switches =====
    parser.add_argument("--use_T", action="store_true", default=True)
    parser.add_argument("--use_R", action="store_true", default=True)
    parser.add_argument("--use_dirichlet", action="store_true", default=True)
    parser.add_argument("--backbone", type=str, default="resnet50", help="vgg16/inception_v3/mobilenet_v2/xception")
    parser.add_argument("--use_T_eval", action="store_true", default=True)
    parser.add_argument("--use_R_eval", action="store_true", default=True)

    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def override_args(args, cli_args):
    if cli_args.epochs is not None:
        args.epochs = cli_args.epochs
    if cli_args.seeds is not None:
        args.seeds = cli_args.seeds
    return args

def build_args_from_config(cfg: dict, cli_args):
    """
    Build args namespace that is fully compatible with current parse_args()
    and train_once / eval_once implementations.
    """

    args = {}

    # ===== required =====
    args["dataset_root"] = cfg.get("dataset_root", cli_args.dataset_root)
    args["work_dir"] = cfg.get("work_dir", cli_args.work_dir)
    args["note"] = cfg.get("note", cli_args.note)
    args["num_classes"] = int(cfg.get("num_classes", 3))
    args["label_map"] = cfg.get("label_map", {"healthy": 0, "stressed": 1, "diseased": 2})

    # ===== switches (train) =====
    args["use_T"] = cfg.get("use_T", cli_args.use_T)
    args["use_R"] = cfg.get("use_R", cli_args.use_R)
    args["use_dirichlet"] = cfg.get("use_dirichlet", cli_args.use_dirichlet)

    # ===== switches (eval) =====
    args["use_T_eval"] = cfg.get("use_T_eval", cli_args.use_T_eval)
    args["use_R_eval"] = cfg.get("use_R_eval", cli_args.use_R_eval)

    # ===== backbone =====
    args["backbone"] = cfg.get("backbone", cli_args.backbone)

    # ===== training hyper-parameters =====
    args["epochs"] = cfg.get("epochs", 50)
    args["lr"] = float(cfg.get("lr", 1e-6))
    args["lambda_conf"] = cfg.get("lambda_conf", 0.6)
    args["lambda_risk"] = cfg.get("lambda_risk", 0.3)
    
    args["train_ratio"] = cfg.get("train_ratio", 0.8)

    args["flush_every"] = cfg.get("flush_every", 50)

    # ===== early stopping =====
    args["es_monitor"] = cfg.get("es_monitor", "macro_f1")
    args["es_mode"] = cfg.get("es_mode", "max")
    args["es_patience"] = cfg.get("es_patience", 5)
    args["es_min_delta"] = float(cfg.get("es_min_delta", 1e-3))
    args["es_min_epoch"] = cfg.get("es_min_epoch", 5)

    # ===== CV =====
    args["seed"] = cfg.get("seed", 35)
    args["seeds"] = cfg.get("seeds", [22, 24, 30, 42, 67])
    # 22 24 30 42 67
    return SimpleNamespace(**args)


def run_cv(args):
    """
    Cross-validation runner.
    train_once: takes args
    eval_once : takes explicit parameters
    """

    os.makedirs(args.work_dir, exist_ok=True)

    seeds: List[int] = args.seeds
    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        print("=" * 100)
        print(f"[CV] Running experiment with seed={seed}")

        # -------------------------
        # 1. prepare args for this run
        # -------------------------
        run_args = copy.deepcopy(args)
        run_args.seed = seed
        run_args.note = f"{args.note}_seed{seed}"

        # -------------------------
        # 2. train once
        # -------------------------
        train_result = train_once(run_args)

        run_dir = train_result["run_dir"]
        best_epoch = train_result.get("best_epoch", None)

        print(f"[CV] Training finished. best_epoch={best_epoch}")
        print(f"[CV] run_dir={run_dir}")

        # -------------------------
        # 3. eval once (best / last model)
        # -------------------------
        vision_ckpt_dir = os.path.join(run_dir, "checkpoints")
        vision_ckpt = os.path.join(vision_ckpt_dir, "visual_best.pth")
        decision_ckpt_dir = os.path.join(run_dir, "checkpoints")
        decision_ckpt = os.path.join(decision_ckpt_dir, "decision_best.pth")
        eval_dir = os.path.join(run_dir, "eval")

        metrics = eval_once(
            dataset_root=run_args.dataset_root,
            vision_ckpt=vision_ckpt,
            decision_ckpt=decision_ckpt,
            seed=seed,
            use_T=getattr(run_args, "use_T_eval", True),
            use_R=getattr(run_args, "use_R_eval", True),
            save_dir=eval_dir,
            label_map=run_args.label_map,
            num_classes=run_args.num_classes,
            backbone=run_args.backbone,
        )

        print(
            f"[CV] seed={seed} "
            f"acc={metrics['accuracy']:.3f} "
            f"macro_f1={metrics['macro_f1']:.3f}"
            f"macro_precision={metrics['macro_precision']:.3f}",
            f"macro_recall={metrics['macro_recall']:.3f}",
            f"healthy_precision={metrics['healthy_precision']:.3f}",
            f"stressed_recall={metrics['stressed_recall']:.3f}",
            f"diseased_recall={metrics['diseased_recall']:.3f}",
        )

        all_results.append(
            {
                "seed": seed,
                "run_dir": run_dir,
                "best_epoch": best_epoch,
                "metrics": metrics,
            }
        )

    # -------------------------
    # 4. aggregate results
    # -------------------------
    accs = [r["metrics"]["accuracy"] for r in all_results]
    macro_f1s = [r["metrics"]["macro_f1"] for r in all_results]
    macro_pre = [r["metrics"]["macro_precision"] for r in all_results]
    macro_rec = [r["metrics"]["macro_recall"] for r in all_results]
    h_pre = [r["metrics"]["healthy_precision"] for r in all_results]
    s_rec = [r["metrics"]["stressed_recall"] for r in all_results]
    d_rec = [r["metrics"]["diseased_recall"] for r in all_results]

    def mean_std(xs):
        mean = sum(xs) / len(xs)
        std = (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5
        return mean, std

    acc_mean, acc_std = mean_std(accs)
    f1_mean, f1_std = mean_std(macro_f1s)
    pre_mean, pre_std = mean_std(macro_pre)
    rec_mean, rec_std = mean_std(macro_rec)
    h_pre_mean, h_pre_std = mean_std(h_pre)
    s_rec_mean,s_rec_std = mean_std(s_rec)
    d_rec_mean, d_rec_std = mean_std(d_rec)

    summary = {
        "n_runs": len(all_results),
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "macro_f1_mean": f1_mean,
        "macro_f1_std": f1_std,
        "macro_precision_mean": pre_mean,
        "macro_precision_std": pre_std,
        "macro_recall_mean": rec_mean,
        "macro_recall_std": rec_std,
        "healthy_precision_mean": h_pre_mean,
        "healthy_precision_std": h_pre_std,
        "stressed_recall_mean": s_rec_mean,
        "stressed_recall_std": s_rec_std,
        "diseased_recall_mean": d_rec_mean,
        "diseased_recall_std": d_rec_std,
        "per_run": all_results,
    }

    summary_path = os.path.join(args.work_dir, f"cv_summary_{args.note}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 100)
    print("[CV] Finished all runs")
    print(f"[CV] Accuracy : {acc_mean:.3f} ± {acc_std:.3f}")
    print(f"[CV] Macro-F1 : {f1_mean:.3f} ± {f1_std:.3f}")
    print(f"[CV] Macro-Precision : {pre_mean:.3f} ± {pre_std:.3f}")
    print(f"[CV] Macro-Recall : {rec_mean:.3f} ± {rec_std:.3f}")
    print(f"[CV] Healthy-Precision : {h_pre_mean:.3f} ± {h_pre_std:.3f}")
    print(f"[CV] Stressed-Recall : {s_rec_mean:.3f} ± {s_rec_std:.3f}")
    print(f"[CV] Diseased-Recall : {d_rec_mean:.3f} ± {d_rec_std:.3f}")
    print(f"[CV] Summary saved to: {summary_path}")

    return summary


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    # 1. parse CLI (only config + overrides)
    cli_args = parse_args()

    # 2. load yaml config
    cfg = load_config(cli_args.config)

    # 3. build args for train_once / eval_once
    args = build_args_from_config(cfg, cli_args)

    # 4. run cross-validation
    results = run_cv(args)
    print("finish: [CV SUMMARY]")
