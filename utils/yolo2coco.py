import os
import json

IMG_W, IMG_H = 2048, 2048

images_dir = "images"
labels_dir = "labels"
output_dir = "annotations"

os.makedirs(output_dir, exist_ok=True)


def yolo_to_xyxy(xc, yc, w, h):
    x1 = (xc - w / 2) * IMG_W
    y1 = (yc - h / 2) * IMG_H
    x2 = (xc + w / 2) * IMG_W
    y2 = (yc + h / 2) * IMG_H
    return [[round(x1, 2), round(y1, 2)],
            [round(x2, 2), round(y2, 2)]]


def infer_mapping(image_name):
    if image_name.startswith("BG-"):
        return {0: "black_gill"}
    if image_name.startswith("WSSV-"):
        return {0: "white_spot"}
    if image_name.startswith("WSSV_BG-"):
        return {0: "white_spot", 1: "black_gill"}
    if image_name.startswith("Healthy-"):
        return {}
    raise ValueError(f"Unknown image prefix: {image_name}")


for img_name in os.listdir(images_dir):
    if not img_name.lower().endswith(".jpg"):
        continue

    label_name = img_name.replace(".jpg", ".txt")
    label_path = os.path.join(labels_dir, label_name)

    mapping = infer_mapping(img_name)

    result = {
        "bbox": None,
        "polygons": {
            "white_spot": []
        },
        "meta": {
            "imagePath": f"..\\images\\{img_name}",
        }
    }

    # Healthy：直接写空
    if not mapping or not os.path.exists(label_path):
        pass
    else:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])

                if cls not in mapping:
                    continue

                lesion_type = mapping[cls]
                bbox = yolo_to_xyxy(xc, yc, w, h)
                result["lesions"][lesion_type].append(bbox)

    out_path = os.path.join(
        output_dir, img_name.replace(".jpg", ".json")
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

print("YOLO → COCO-style JSON 转换完成")
