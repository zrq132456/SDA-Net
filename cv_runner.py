import os
import math
import random
from collections import defaultdict
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
    args["seed"] = cfg.get("seed", 35)  # training reproducibility seed
    args["cv_seed"] = cfg.get("cv_seed", 42)  # fold partition seed
    args["n_folds"] = cfg.get("n_folds", 5)
    return SimpleNamespace(**args)

def collect_ids_and_labels(dataset_root, image_dir="images", converted_dir="raw_converted"):
    """
    Collect sample ids and labels from T.json.
    Used to build stratified 5-fold splits.
    """
    image_path = os.path.join(dataset_root, image_dir)
    converted_path = os.path.join(dataset_root, converted_dir)

    image_exts = (".jpg", ".png", ".jpeg")
    sample_ids = []

    for fname in os.listdir(image_path):
        if fname.lower().endswith(image_exts):
            sample_ids.append(os.path.splitext(fname)[0])

    sample_ids.sort()

    labels = {}
    for sid in sample_ids:
        t_path = os.path.join(converted_path, sid, "T.json")
        if not os.path.exists(t_path):
            raise FileNotFoundError(f"Missing T.json for sample: {sid}")

        with open(t_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)

        label = t_data.get("label", None)
        if label is None:
            raise ValueError(f"Missing label in T.json for sample: {sid}")

        labels[sid] = label

    return sample_ids, labels


def make_stratified_kfold_ids(sample_ids, labels, n_folds=5, seed=42):
    """
    Build stratified k-fold ids.
    Each fold has approximately the same class distribution.
    """
    label_to_ids = defaultdict(list)

    for sid in sample_ids:
        label_to_ids[labels[sid]].append(sid)

    rng = random.Random(seed)
    folds = [[] for _ in range(n_folds)]

    for label, ids in label_to_ids.items():
        ids = list(ids)
        rng.shuffle(ids)

        for i, sid in enumerate(ids):
            folds[i % n_folds].append(sid)

    split_list = []

    all_ids_set = set(sample_ids)
    for fold_idx in range(n_folds):
        test_ids = sorted(folds[fold_idx])
        test_set = set(test_ids)
        train_ids = sorted(list(all_ids_set - test_set))

        split_list.append({
            "fold": fold_idx,
            "train_ids": train_ids,
            "test_ids": test_ids,
        })

    return split_list

def run_cv(args):
    """
    Cross-validation runner.
    train_once: takes args
    eval_once : takes explicit parameters
    """

    os.makedirs(args.work_dir, exist_ok=True)

    sample_ids, labels = collect_ids_and_labels(args.dataset_root)
    fold_splits = make_stratified_kfold_ids(
        sample_ids=sample_ids,
        labels=labels,
        n_folds=args.n_folds,
        seed=args.cv_seed,
    )

    all_results: List[Dict[str, Any]] = []
    all_records: List[Dict[str, Any]] = []

    for fold_info in fold_splits:

        fold = fold_info["fold"]

        train_ids_pool = list(fold_info["train_ids"])
        test_ids = fold_info["test_ids"]

        rng = random.Random(args.seed + fold)
        rng.shuffle(train_ids_pool)

        n_val = max(1, math.ceil(len(train_ids_pool) * 0.2))

        val_ids = train_ids_pool[:n_val]
        real_train_ids = train_ids_pool[n_val:]

        print("=" * 100)
        print(f"[CV] Running fold={fold + 1}/{args.n_folds}")

        print(
            f"[CV] train={len(real_train_ids)}, "
            f"val={len(val_ids)}, "
            f"test={len(test_ids)}"
        )

        run_args = copy.deepcopy(args)
        run_args.fold = fold
        run_args.seed = args.seed + fold

        run_args.note = f"{args.note}_fold{fold}"

        run_args.train_ids = real_train_ids
        run_args.val_ids = val_ids
        run_args.test_ids = test_ids

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

        eval_result = eval_once(
            dataset_root=run_args.dataset_root,
            vision_ckpt=vision_ckpt,
            decision_ckpt=decision_ckpt,
            seed=run_args.seed,
            ids=run_args.test_ids,
            use_T=getattr(run_args, "use_T_eval", True),
            use_R=getattr(run_args, "use_R_eval", True),
            save_dir=eval_dir,
            label_map=run_args.label_map,
            num_classes=run_args.num_classes,
            backbone=run_args.backbone,
        )
        metrics = eval_result["metrics"]
        records = eval_result["records"]

        # print(
        #     f"[CV] seed={seed} "
        #     f"acc={metrics['accuracy']:.3f} "
        #     f"macro_f1={metrics['macro_f1']:.3f}"
        #     f"macro_precision={metrics['macro_precision']:.3f}",
        #     f"macro_recall={metrics['macro_recall']:.3f}",
        #     f"healthy_precision={metrics['healthy_precision']:.3f}",
        #     f"stressed_recall={metrics['stressed_recall']:.3f}",
        #     f"diseased_recall={metrics['diseased_recall']:.3f}",
        # )

        # per_class_f1_all = [r["metrics"]["per_class_f1"] for r in all_results]

        print(
            f"[CV] fold={fold + 1} "
            f"acc={metrics['accuracy']:.3f} "
            f"macro_f1={metrics['macro_f1']:.3f} "
            f"macro_precision={metrics['macro_precision']:.3f} "
            f"macro_recall={metrics['macro_recall']:.3f} "
            # f"Healthy_precision={metrics['healthy_precision']:.3f} "
            # f"Diseased_recall={metrics['diseased_recall']:.3f}"
            f"per_class_f1: \n{metrics['per_class_f1']}\n"
        )

        for r in records:
            rr = copy.deepcopy(r)

            rr["fold"] = fold
            rr["seed"] = run_args.seed
            rr["run_dir"] = run_dir
            rr["best_epoch"] = best_epoch

            all_records.append(rr)

        all_results.append(
            {
                "fold": fold,
                "seed": run_args.seed,
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
    # h_pre = [r["metrics"]["healthy_precision"] for r in all_results]
    # s_rec = [r["metrics"]["stressed_recall"] for r in all_results]
    # d_rec = [r["metrics"]["diseased_recall"] for r in all_results]
    # h_pre = [r["metrics"]["healthy_precision"] for r in all_results]
    # d_rec = [r["metrics"]["diseased_recall"] for r in all_results]

    # 统计每类 F1
    per_class_keys = list(all_results[0]["metrics"]["per_class_f1"].keys())
    per_class_vals = {k: [r["metrics"]["per_class_f1"][k] for r in all_results] for k in per_class_keys}

    def mean_std(xs):
        mean = sum(xs) / len(xs)
        std = (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5
        return mean, std

    acc_mean, acc_std = mean_std(accs)
    f1_mean, f1_std = mean_std(macro_f1s)
    pre_mean, pre_std = mean_std(macro_pre)
    rec_mean, rec_std = mean_std(macro_rec)
    # h_pre_mean, h_pre_std = mean_std(h_pre)
    # s_rec_mean,s_rec_std = mean_std(s_rec)
    # d_rec_mean, d_rec_std = mean_std(d_rec)

    # per-class mean/std
    per_class_mean_std = {}
    for cls in per_class_keys:
        mean, std = mean_std(per_class_vals[cls])
        per_class_mean_std[cls] = {"mean": mean, "std": std}

    # summary = {
    #     "n_runs": len(all_results),
    #     "accuracy_mean": acc_mean,
    #     "accuracy_std": acc_std,
    #     "macro_f1_mean": f1_mean,
    #     "macro_f1_std": f1_std,
    #     "macro_precision_mean": pre_mean,
    #     "macro_precision_std": pre_std,
    #     "macro_recall_mean": rec_mean,
    #     "macro_recall_std": rec_std,
    #     "healthy_precision_mean": h_pre_mean,
    #     "healthy_precision_std": h_pre_std,
    #     "stressed_recall_mean": s_rec_mean,
    #     "stressed_recall_std": s_rec_std,
    #     "diseased_recall_mean": d_rec_mean,
    #     "diseased_recall_std": d_rec_std,
    #     "per_run": all_results,
    # }
    # summary = {
    #     "n_runs": len(all_results),
    #
    #     "accuracy_mean": acc_mean,
    #     "accuracy_std": acc_std,
    #
    #     "macro_f1_mean": f1_mean,
    #     "macro_f1_std": f1_std,
    #
    #     "macro_precision_mean": pre_mean,
    #     "macro_precision_std": pre_std,
    #
    #     "macro_recall_mean": rec_mean,
    #     "macro_recall_std": rec_std,
    #
    #     "Healthy_precision_mean": h_pre_mean,
    #     "Healthy_precision_std": h_pre_std,
    #
    #     "Diseased_recall_mean": d_rec_mean,
    #     "Diseased_recall_std": d_rec_std,
    #
    #     "per_run": all_results,
    # }

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
        "per_class_f1_mean_std": per_class_mean_std,
        "per_run": all_results,
    }

    summary_path = os.path.join(args.work_dir, f"cv_summary_{args.note}.json")
    records_path = os.path.join(args.work_dir, f"cv_records_{args.note}.jsonl")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(records_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    print("=" * 100)
    print("[CV] Finished all runs")
    print(f"[CV] Accuracy : {acc_mean:.3f} ± {acc_std:.3f}")
    print(f"[CV] Macro-F1 : {f1_mean:.3f} ± {f1_std:.3f}")
    print(f"[CV] Macro-Precision : {pre_mean:.3f} ± {pre_std:.3f}")
    print(f"[CV] Macro-Recall : {rec_mean:.3f} ± {rec_std:.3f}")
    # print(f"[CV] Healthy-Precision : {h_pre_mean:.3f} ± {h_pre_std:.3f}")
    # print(f"[CV] Stressed-Recall : {s_rec_mean:.3f} ± {s_rec_std:.3f}")
    # print(f"[CV] Diseased-Recall : {d_rec_mean:.3f} ± {d_rec_std:.3f}")

    # ------------------------
    # Print per-class F1 dynamically
    # ------------------------
    print("\n[CV] Per-class F1:")
    for cls, stats in summary["per_class_f1_mean_std"].items():
        print(f"[CV] {cls} : {stats['mean']:.3f} ± {stats['std']:.3f}")

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
