import os
import random
import numpy as np
import torch


# =====================================================
# 1. Global seed
# =====================================================

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # cpu only, but still keep these
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================
# 2. Load all sample ids (single source of truth)
# =====================================================

def load_all_ids(
    dataset_root: str = "../datasets",
    image_dir: str = "images",
    exts=(".jpg", ".png", ".jpeg"),
):
    img_dir = os.path.join(dataset_root, image_dir)
    ids = []

    for fn in os.listdir(img_dir):
        if fn.lower().endswith(exts):
            ids.append(os.path.splitext(fn)[0])

    ids.sort()
    return ids


# =====================================================
# 3. Deterministic train / test split
# =====================================================

def split_ids(
    all_ids,
    train_ratio=0.8,
    seed=42,
):
    rng = random.Random(seed)
    ids = list(all_ids)
    rng.shuffle(ids)

    n_train = int(len(ids) * train_ratio)
    train_ids = ids[:n_train]
    test_ids = ids[n_train:]

    return train_ids, test_ids
