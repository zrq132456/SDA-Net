import os
import cv2
import shutil
import numpy as np

from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim


# =====================================================
# Config
# =====================================================
set_all = [
    "Bacterial_Aeromoniasis", "Bacterial_gill_disease", "Bacterial_Red_disease",
    "Fungal_diseases_Saprolegniosis", "Healthy_Fish", "Parasitic_diseases", "Viral_diseases_White_tail_disease"
]


# Load image
def load_gray(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img


# =====================================================
# Union Find
# =====================================================
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        pa = self.find(a)
        pb = self.find(b)
        if pa != pb:
            self.parent[pb] = pa


# Main
for i in tqdm(range(len(set_all))):
    set_name = set_all[i]
    DATA_DIR = f"./{set_name}"
    DUP_DIR = f"./{set_name}_duplicates"
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    SSIM_THRESHOLD = 0.97
    IMG_SIZE = 256

    image_paths = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                image_paths.append(os.path.join(root, f))
    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images")

    # Read images
    images = []
    for path in tqdm(image_paths, desc="Loading"):
        img = load_gray(path)
        images.append(img)

    # Compare all pairs
    uf = UnionFind(len(images))
    duplicate_pairs = []
    print("Computing SSIM ...")
    for i in tqdm(range(len(images))):
        img1 = images[i]
        if img1 is None:
            continue
        for j in range(i + 1, len(images)):
            img2 = images[j]
            if img2 is None:
                continue
            score = ssim(img1, img2)
            if score >= SSIM_THRESHOLD:
                uf.union(i, j)
                duplicate_pairs.append((image_paths[i], image_paths[j], score))

    # Build clusters
    clusters = {}
    for i in range(len(images)):
        root = uf.find(i)
        clusters.setdefault(root, []).append(i)

    duplicate_groups = [g for g in clusters.values() if len(g) > 1]
    print(f"Duplicate groups: {len(duplicate_groups)}")

    # Move duplicates
    os.makedirs(DUP_DIR, exist_ok=True)
    moved = 0
    report_lines = []
    for gid, group in enumerate(duplicate_groups):
        keep_idx = group[0]
        keep_path = image_paths[keep_idx]
        report_lines.append(f"\n===== GROUP {gid+1} =====")
        report_lines.append(f"KEEP: {keep_path}")

        for idx in group[1:]:
            dup_path = image_paths[idx]
            score = ssim(images[keep_idx], images[idx])
            report_lines.append(f"DUP : {dup_path}")
            report_lines.append(f"SSIM: {score:.4f}")
            dst = os.path.join(DUP_DIR, os.path.basename(dup_path))
            name, ext = os.path.splitext(dst)
            k = 1
            while os.path.exists(dst):
                dst = f"{name}_{k}{ext}"
                k += 1
            shutil.move(dup_path, dst)
            moved += 1

    # Save report
    with open(f"{set_name}_duplicate_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Total Images: {len(image_paths)}\n")
        f.write(f"Duplicate Groups: {len(duplicate_groups)}\n")
        f.write(f"Moved Images: {moved}\n\n")
        f.write("\n".join(report_lines))

    print("=" * 60)
    print("Finished")
    print("Duplicate Groups :", len(duplicate_groups))
    print("Moved Images     :", moved)
    print("Report           : duplicate_report.txt")
    print("=" * 60)
