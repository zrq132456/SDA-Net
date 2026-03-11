import os
import json
from typing import Dict, Any, List
import random

from PIL import Image
from torch.utils.data import Dataset

class ShrimpDataset(Dataset):
    def __init__(
        self,
        dataset_root: str = "../datasets",
        image_dir: str = "images",
        converted_dir: str = "raw_converted",
        image_exts: List[str] = (".jpg", ".png", ".jpeg"),
        load_image: bool = False,
        use_bbox = True,
        use_region = True,
        use_lesion = True,
        transform = None,
        ids = None,
        split: str = None,  # "train"/"test"/None
        split_ratio: float = 0.8,
        split_seed: int = 42,
    ):
        self.dataset_root = dataset_root
        self.image_dir = os.path.join(dataset_root, image_dir)
        self.converted_dir = os.path.join(dataset_root, converted_dir)
        self.image_exts = image_exts
        self.load_image = load_image
        self.use_bbox = use_bbox
        self.use_region = use_region
        self.use_lesion = use_lesion
        self.transform = transform

        # collect all ids once
        self.all_sample_ids = self._collect_sample_ids()
        # decide active ids
        if ids is not None:
            self.active_ids = list(ids)
        else:
            self.active_ids = self._make_split(
                split=split,
                ratio=split_ratio,
                seed=split_seed,
            )

    def __len__(self):
        # return len(self.sample_ids)
        return len(self.active_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id = self.active_ids[idx]

        sample = {
            "id": sample_id,
            "label": None,    # optional, train only
            "T_explicit": None,        # optional
            "R_explicit": None,        # optional
            "I_explicit": None,
        }

        # ---------- Load T ----------
        t_path = os.path.join(self.converted_dir, sample_id, "T.json")
        if os.path.exists(t_path):
            with open(t_path, "r", encoding="utf-8") as f:
                t_data = json.load(f)
            sample["id"] = t_data.get("id", None)
            sample["label"] = t_data.get("label", None)

            # -------- Normalize T --------
            sample["T_explicit"] = {
                "Angle": t_data.get("T#Angle", None),
                "Visible_flags": {
                    "head": t_data.get("T#head", None),
                    "body": t_data.get("T#body", None),
                    "tail": t_data.get("T#tail", None),
                    "antenna": t_data.get("T#antenna", None),
                    "limb": t_data.get("T#limb", None),
                },
                "Spot": {
                    "white_spot": t_data.get("T#white_spot", None),
                    "black_spot": t_data.get("T#black_spot", None),
                    "speckle": t_data.get("T#speckle", None),
                },
                "Trauma": {
                    "eye_loss": t_data.get("T#eye_loss", None),
                    "antenna_loss": t_data.get("T#antenna_loss", None),
                    "limb_loss": t_data.get("T#limb_loss", None),
                    "shell_crack": t_data.get("T#shell_crack", None),
                },
                "Respiratory": {
                    "gill_color": t_data.get("T#gill_color", None),
                    "gill_dirty": t_data.get("T#gill_dirty", None),
                },
                "Surface": {
                    "eye_color": t_data.get("T#eye_color", None),
                    "tail_red": t_data.get("T#tail_red", None),
                    "limb_red": t_data.get("T#limb_red", None),
                    "limb_black": t_data.get("T#limb_black", None),
                },
                "Digest": {
                    "gut_empty": t_data.get("T#gut_empty", None),
                    "gut_white": t_data.get("T#gut_white", None),
                    "hp_shape": t_data.get("T#hp_shape", None),
                    "hp_color": t_data.get("T#hp_color", None),
                    "stomach_atrophy": t_data.get("T#stomach_atrophy", None),
                },
                "Body": {
                    "shell_color": t_data.get("T#shell_color", None),
                    "opacity": t_data.get("T#opacity", None),
                    "morphology": t_data.get("T#morphology", None),
                }
            }

        # ---------- Load R ----------
        r_path = os.path.join(self.converted_dir, sample_id, "R.json")
        if os.path.exists(r_path):
            with open(r_path, "r", encoding="utf-8") as f:
                r_data = json.load(f)

            # -------- Normalize R --------
            sample["R_explicit"] = {
                "angle": r_data.get("angle", None),
                "organ_ratio": {
                    "hp": r_data.get("R#hp", None),
                    "stomach": r_data.get("R#stomach", None),
                },
                "lesion_stats": {
                    "white_spot_match": r_data.get("R#white_spot", None),
                    "black_spot_match": r_data.get("R#black_spot", None),
                    "speckle_ratio": r_data.get("R#speckle", None),
                    "red_tail_ratio": r_data.get("R#red_tail", None),
                    "opacity_ratio": r_data.get("R#opacity", None),
                },
                "morphological_stats": {
                    "curvature_rate": r_data.get("R#curve", None),
                    "bend_degree": r_data.get("R#bend", None),
                },
                "derived_from_unknown_T": 0,
            }

        # ---------- Load I ----------
        I_REGION_KEYS = ["hp", "gill", "eye", "head", "stomach", "tail", "helmet", "body"]
        I_LESION_KEYS = ["white_spot", "black_spot", "red_tail", "opacity", "speckle"]
        i_path = os.path.join(self.converted_dir, sample_id, "I.json")
        if os.path.exists(i_path):
            with open(i_path, "r", encoding="utf-8") as f:
                i_data = json.load(f)

            # -------- Normalize I --------
            regions = {k: {"polygons": [], "count": 0} for k in I_REGION_KEYS}
            if self.use_region:
                for region_ in I_REGION_KEYS:
                    regions[region_]["polygons"] = i_data["polygons"].get(region_, [])
                    regions[region_]["count"] = len(i_data["polygons"].get(region_, []))

            lesions = {k: {"polygons": [], "count": 0} for k in I_LESION_KEYS}
            if self.use_lesion:
                for lesions_ in I_LESION_KEYS:
                    lesions[lesions_]["polygons"] = i_data["polygons"].get(lesions_, [])
                    lesions[lesions_]["count"] = len(i_data["polygons"].get(lesions_, []))

            bbox = i_data["bbox"] if self.use_bbox else None

            sample["I_explicit"] = {
                "image": self._load_image(sample_id),      # path/loaded image
                "bbox": i_data["bbox"],
                "regions": regions,
                "lesions": lesions,
                "source": "annotation"
            }

        return sample

    # ----------------
    def _collect_sample_ids(self) -> List[str]:
        """
        Collect sample ids based on image directory.
        Image is treated as mandatory modality.
        """
        ids = []
        for fname in os.listdir(self.image_dir):
            if fname.lower().endswith(self.image_exts):
                ids.append(os.path.splitext(fname)[0])
        ids.sort()
        return ids

    def _load_image(self, sample_id: str):
        """
        Load image or return image path.
        """
        for ext in self.image_exts:
            img_path = os.path.join(self.image_dir, sample_id + ext)
            if os.path.exists(img_path):
                if not self.load_image:
                    return img_path

                img = Image.open(img_path).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                return img

        raise FileNotFoundError(f"No image found for sample {sample_id}")

    def _make_split(self, split, ratio, seed):
        """
        Return a list of sample_ids according to split config.
        """
        if split is None:
            return self.all_sample_ids

        assert split in ["train", "test"]
        assert 0.0 < ratio < 1.0

        ids = list(self.all_sample_ids)
        random.Random(seed).shuffle(ids)

        n_train = int(len(ids) * ratio)

        if split == "train":
            return ids[:n_train]
        else:
            return ids[n_train:]
