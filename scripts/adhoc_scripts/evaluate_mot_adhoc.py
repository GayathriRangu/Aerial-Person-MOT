import os
import pandas as pd
import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a: np.asarray(a, dtype=float)

import motmetrics as mm

# ==========================================
# CONFIG
# ==========================================
EXPERIMENT_NAME = "E02_Highres_Baseline"
SEQ_NAME = "uav0000086_00000_v"
log_file = "../results/experiment_log.csv"
GT_FILE = (
    f"../data/VisDrone2019-MOT-val/"
    f"annotations/{SEQ_NAME}.txt"
)

TRACK_FILE = (
    f"../tracker_outputs/{SEQ_NAME}.txt"
)

# ==========================================
# LOAD FILES
# ==========================================

gt_cols = [
    "frame",
    "id",
    "x",
    "y",
    "w",
    "h",
    "score",
    "class",
    "trunc",
    "occ"
]

pred_cols = [
    "frame",
    "id",
    "x",
    "y",
    "w",
    "h",
    "score",
    "tmp1",
    "tmp2",
    "tmp3"
]

gt = pd.read_csv(
    GT_FILE,
    header=None,
    names=gt_cols
)

pred = pd.read_csv(
    TRACK_FILE,
    header=None,
    names=pred_cols
)

# ==========================================
# KEEP ONLY PERSON CLASSES
# ==========================================

gt = gt[
    gt["class"].isin([1, 2])
]

# ==========================================
# ACCUMULATOR
# ==========================================

acc = mm.MOTAccumulator(auto_id=True)

frames = sorted(
    gt["frame"].unique()
)

# ==========================================
# FRAME LOOP
# ==========================================

for frame_id in frames:

    gt_frame = gt[
        gt["frame"] == frame_id
    ]

    pred_frame = pred[
        pred["frame"] == frame_id
    ]

    gt_ids = gt_frame["id"].tolist()

    pred_ids = pred_frame["id"].tolist()

    gt_boxes = gt_frame[
        ["x", "y", "w", "h"]
    ].values

    pred_boxes = pred_frame[
        ["x", "y", "w", "h"]
    ].values

    # --------------------------------------
    # IoU Distance Matrix
    # --------------------------------------

    distances = mm.distances.iou_matrix(
        gt_boxes,
        pred_boxes,
        max_iou=0.5
    )

    acc.update(
        gt_ids,
        pred_ids,
        distances
    )

# ==========================================
# METRICS
# ==========================================

mh = mm.metrics.create()

summary = mh.compute(
    acc,
    metrics=[
        "num_frames",
        "mota",
        "motp",
        "idf1",
        "precision",
        "recall",
        "num_switches",
        "num_false_positives",
        "num_misses"
    ],
    name="ByteTrack"
)

print("\n")
print("=" * 60)
print("MOT EVALUATION")
print("=" * 60)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(summary.to_string())

print("=" * 60)

row = {
    "Experiment": EXPERIMENT_NAME,
    "Frames": summary.loc["ByteTrack","num_frames"],
    "MOTA": summary.loc["ByteTrack","mota"],
    "MOTP": summary.loc["ByteTrack","motp"],
    "IDF1": summary.loc["ByteTrack","idf1"],
    "Precision": summary.loc["ByteTrack","precision"],
    "Recall": summary.loc["ByteTrack","recall"],
    "IDSW": summary.loc["ByteTrack","num_switches"],
    "FP": summary.loc["ByteTrack","num_false_positives"],
    "FN": summary.loc["ByteTrack","num_misses"]
}

df = pd.DataFrame([row])

if os.path.exists(log_file):
    df.to_csv(
        log_file,
        mode="a",
        header=False,
        index=False
    )
else:
    df.to_csv(
        log_file,
        index=False
    )