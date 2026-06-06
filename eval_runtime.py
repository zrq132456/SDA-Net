import os
import json
import time
import argparse
import platform
from typing import Dict, Any, List

import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset.dataloader import ShrimpDataset
from dataset.utils import set_global_seed, load_all_ids, split_ids

from models.backbone.backbone_multiscale import VisualEvidenceNet
from models.fusion.t_branch.t_process import TProcess
from models.fusion.r_branch.r_process import RProcess
from models.fusion.decision_module import DecisionModuleV7


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


def sync_if_cuda(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def summarize_time(times):
    arr = np.array(times, dtype=np.float64)
    return {
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std()),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "mean_ms": float(arr.mean() * 1000),
        "fps": float(1.0 / arr.mean()) if arr.mean() > 0 else 0.0,
    }


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def eval_runtime(
    dataset_root: str,
    vision_ckpt: str,
    decision_ckpt: str,
    seed: int = 42,
    use_T: bool = True,
    use_R: bool = True,
    label_map: Dict[str, int] = None,
    num_classes: int = 3,
    backbone: str = "resnet50",
    warmup: int = 5,
    save_dir: str = None,
):
    set_global_seed(seed)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    LABEL_MAP = label_map
    INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
    K = num_classes
    ALL_NAMES = [INV_LABEL_MAP[i] for i in range(K)]

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

    vision = VisualEvidenceNet(
        num_classes=K,
        pretrained=False,
        backbone=backbone
    ).to(device)

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

    timing_records: List[Dict[str, Any]] = []

    valid_samples = []
    for sample in loader:
        label_str = sample.get("label", None)

        if label_str not in LABEL_MAP:
            continue
        if sample.get("I_explicit") is None:
            continue

        valid_samples.append(sample)

    if len(valid_samples) == 0:
        raise RuntimeError("No valid samples found. Please check dataset_root and annotations.")

    print(f"[INFO] Valid samples: {len(valid_samples)}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Backbone: {backbone}")

    # -------------------------
    # Warmup
    # -------------------------
    with torch.no_grad():
        for sample in valid_samples[:min(warmup, len(valid_samples))]:
            I_out = vision(sample["I_explicit"])
            e = I_out["e"]

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

            _ = decision(e=e, T_out=T_out, R_out=R_out)

        sync_if_cuda(device)

    # -------------------------
    # Runtime evaluation
    # -------------------------
    with torch.no_grad():
        for sample in valid_samples:
            sample_id = sample.get("id", None)

            sync_if_cuda(device)
            full_start = time.perf_counter()

            # I branch
            sync_if_cuda(device)
            t0 = time.perf_counter()
            I_out = vision(sample["I_explicit"])
            sync_if_cuda(device)
            t1 = time.perf_counter()
            I_time = t1 - t0

            e = I_out["e"]

            # T branch
            sync_if_cuda(device)
            t0 = time.perf_counter()
            T_out = (
                T_proc(sample.get("T_explicit"))
                if (use_T and sample.get("T_explicit") is not None)
                else None
            )
            sync_if_cuda(device)
            t1 = time.perf_counter()
            T_time = t1 - t0

            # R branch
            sync_if_cuda(device)
            t0 = time.perf_counter()
            R_out = (
                R_proc(sample.get("R_explicit"))
                if (use_R and sample.get("R_explicit") is not None)
                else None
            )
            sync_if_cuda(device)
            t1 = time.perf_counter()
            R_time = t1 - t0

            # decision module
            sync_if_cuda(device)
            t0 = time.perf_counter()
            out = decision(e=e, T_out=T_out, R_out=R_out)
            sync_if_cuda(device)
            t1 = time.perf_counter()
            decision_time = t1 - t0

            P = out["P"]
            pred = int(P.argmax().item())

            sync_if_cuda(device)
            full_end = time.perf_counter()

            sdanet_time = I_time + T_time + R_time + decision_time
            full_time = full_end - full_start

            timing_records.append({
                "id": sample_id,
                "pred_idx": pred,
                "I_branch_s": I_time,
                "T_branch_s": T_time,
                "R_branch_s": R_time,
                "decision_s": decision_time,
                "sdanet_inference_s": sdanet_time,
                "full_pipeline_s": full_time,
            })

    keys = [
        "I_branch_s",
        "T_branch_s",
        "R_branch_s",
        "decision_s",
        "sdanet_inference_s",
        "full_pipeline_s",
    ]

    summary = {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "model": {
            "backbone": backbone,
            "num_classes": K,
            "use_T": use_T,
            "use_R": use_R,
            "vision_params": count_params(vision),
            "decision_params": count_params(decision),
            "total_params": count_params(vision) + count_params(decision),
        },
        "data": {
            "dataset_root": dataset_root,
            "num_valid_samples": len(valid_samples),
            "seed": seed,
            "split": "validation split, train_ratio=0.8",
            "warmup": warmup,
        },
        "runtime": {
            k: summarize_time([r[k] for r in timing_records])
            for k in keys
        }
    }

    print("\n========== Runtime Summary ==========")
    for k in keys:
        item = summary["runtime"][k]
        print(
            f"{k:22s}: "
            f"{item['mean_ms']:.3f} ± {item['std_s'] * 1000:.3f} ms/image, "
            f"FPS = {item['fps']:.2f}"
        )

    print("\n========== Hardware ==========")
    for k, v in summary["hardware"].items():
        print(f"{k}: {v}")

    print("\n========== Model ==========")
    print(f"Vision params:   {summary['model']['vision_params']:,}")
    print(f"Decision params: {summary['model']['decision_params']:,}")
    print(f"Total params:    {summary['model']['total_params']:,}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "runtime_records.jsonl"), "w", encoding="utf-8") as f:
            for r in timing_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with open(os.path.join(save_dir, "runtime_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n[INFO] Runtime results saved to: {save_dir}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--vision_ckpt", type=str, required=True)
    parser.add_argument("--decision_ckpt", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--num_classes", type=int, default=3)

    parser.add_argument("--task", type=str, default="state", choices=["state", "disease"])

    parser.add_argument("--use_T", action="store_true", default=True)
    parser.add_argument("--use_R", action="store_true", default=True)

    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="runtime_eval")

    args = parser.parse_args()

    if args.task == "state":
        label_map = {
            "healthy": 0,
            "stressed": 1,
            "diseased": 2,
        }
        num_classes = 3
    else:
        label_map = {
            "healthy": 0,
            "wssv": 1,
            "bg": 2,
            "wssv_bg": 3,
        }
        num_classes = 4

    eval_runtime(
        dataset_root=args.dataset_root,
        vision_ckpt=args.vision_ckpt,
        decision_ckpt=args.decision_ckpt,
        seed=args.seed,
        use_T=args.use_T,
        use_R=args.use_R,
        label_map=label_map,
        num_classes=num_classes,
        backbone=args.backbone,
        warmup=args.warmup,
        save_dir=args.save_dir,
    )
