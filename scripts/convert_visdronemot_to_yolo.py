import os
import cv2
from collections import defaultdict

# =====================================================
# CONFIG
# =====================================================

# DATASET_ROOT = "../data/VisDrone2019-MOT-train"
DATASET_ROOT = "../data/VisDrone2019-MOT-val"
SEQUENCES_DIR = os.path.join(
    DATASET_ROOT,
    "sequences"
)

ANNOTATIONS_DIR = os.path.join(
    DATASET_ROOT,
    "annotations"
)

YOLO_LABEL_DIR = os.path.join(
    DATASET_ROOT,
    "yolo_labels"
)

os.makedirs(
    YOLO_LABEL_DIR,
    exist_ok=True
)

VALID_CLASSES = [1, 2]

CLASS_MAP = {
    1: 0,
    2: 1
}

# =====================================================
# CONVERSION
# =====================================================

for ann_file in os.listdir(ANNOTATIONS_DIR):

    if not ann_file.endswith(".txt"):
        continue

    seq_name = ann_file.replace(
        ".txt",
        ""
    )

    print(f"Processing {seq_name}")

    seq_dir = os.path.join(
        SEQUENCES_DIR,
        seq_name
    )

    out_seq_dir = os.path.join(
        YOLO_LABEL_DIR,
        seq_name
    )

    os.makedirs(
        out_seq_dir,
        exist_ok=True
    )

    labels_per_frame = defaultdict(list)

    ann_path = os.path.join(
        ANNOTATIONS_DIR,
        ann_file
    )

    with open(ann_path) as f:

        for line in f:

            vals = line.strip().split(",")

            frame_id = int(vals[0])

            x = float(vals[2])
            y = float(vals[3])
            w = float(vals[4])
            h = float(vals[5])

            cls = int(vals[7])

            if cls not in VALID_CLASSES:
                continue

            img_name = f"{frame_id:07d}.jpg"

            img_path = os.path.join(
                seq_dir,
                img_name
            )

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)

            H, W = img.shape[:2]

            xc = (x + w/2) / W
            yc = (y + h/2) / H

            wn = w / W
            hn = h / H

            labels_per_frame[
                img_name
            ].append(
                f"{CLASS_MAP[cls]} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{wn:.6f} "
                f"{hn:.6f}\n"
            )

    # ---------------------------------------
    # Save labels
    # ---------------------------------------

    for img_name, labels in labels_per_frame.items():

        label_file = os.path.join(
            out_seq_dir,
            img_name.replace(
                ".jpg",
                ".txt"
            )
        )

        with open(label_file, "w") as f:
            f.writelines(labels)

print("\nDONE")