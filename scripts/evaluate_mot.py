import os
import pandas as pd
import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a: np.asarray(a, dtype=float)

import motmetrics as mm

# =====================================================
# CONFIG
# =====================================================

EXPERIMENT_NAME = "E01_Baseline"

GT_DIR = "../data/VisDrone2019-MOT-val/annotations"

PRED_DIR = (
    "../outputs/"
    "E01_baseline/"
    "tracker_outputs"
)

RESULTS_DIR = "../results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

PER_SEQUENCE_FILE = os.path.join(
    RESULTS_DIR,
    f"{EXPERIMENT_NAME}_per_sequence.csv"
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    f"{EXPERIMENT_NAME}_summary.csv"
)

# =====================================================
# COLUMN NAMES
# =====================================================

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

# =====================================================
# STORAGE
# =====================================================

all_results = []

overall_acc = mm.MOTAccumulator(
    auto_id=True
)

# =====================================================
# LOOP OVER ALL SEQUENCES
# =====================================================

prediction_files = sorted(
    [
        f for f in os.listdir(PRED_DIR)
        if f.endswith(".txt")
    ]
)

for pred_file in prediction_files:

    seq_name = pred_file.replace(
        ".txt",
        ""
    )

    gt_file = os.path.join(
        GT_DIR,
        pred_file
    )

    pred_file_path = os.path.join(
        PRED_DIR,
        pred_file
    )

    if not os.path.exists(gt_file):

        print(
            f"GT missing for {seq_name}"
        )

        continue

    print(
        f"Evaluating {seq_name}"
    )

    gt = pd.read_csv(
        gt_file,
        header=None,
        names=gt_cols
    )

    pred = pd.read_csv(
        pred_file_path,
        header=None,
        names=pred_cols
    )

    # ==========================================
    # KEEP ONLY PERSON CLASSES
    # ==========================================

    gt = gt[
        gt["class"].isin([1, 2])
    ]

    acc = mm.MOTAccumulator(
        auto_id=True
    )

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

        gt_ids = gt_frame[
            "id"
        ].tolist()

        pred_ids = pred_frame[
            "id"
        ].tolist()

        gt_boxes = gt_frame[
            ["x", "y", "w", "h"]
        ].values

        pred_boxes = pred_frame[
            ["x", "y", "w", "h"]
        ].values

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

        overall_acc.update(
            gt_ids,
            pred_ids,
            distances
        )

    # ==========================================
    # METRICS FOR THIS VIDEO
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
        name=seq_name
    )

    row = {

        "Sequence": seq_name,

        "Frames":
        summary.loc[
            seq_name,
            "num_frames"
        ],

        "MOTA":
        summary.loc[
            seq_name,
            "mota"
        ],

        "MOTP":
        summary.loc[
            seq_name,
            "motp"
        ],

        "IDF1":
        summary.loc[
            seq_name,
            "idf1"
        ],

        "Precision":
        summary.loc[
            seq_name,
            "precision"
        ],

        "Recall":
        summary.loc[
            seq_name,
            "recall"
        ],

        "IDSW":
        summary.loc[
            seq_name,
            "num_switches"
        ],

        "FP":
        summary.loc[
            seq_name,
            "num_false_positives"
        ],

        "FN":
        summary.loc[
            seq_name,
            "num_misses"
        ]
    }

    all_results.append(row)

# =====================================================
# SAVE PER-SEQUENCE CSV
# =====================================================

per_seq_df = pd.DataFrame(
    all_results
)

per_seq_df.to_csv(
    PER_SEQUENCE_FILE,
    index=False
)

# =====================================================
# DATASET LEVEL METRICS
# =====================================================

mh = mm.metrics.create()

overall = mh.compute(
    overall_acc,
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
    name=EXPERIMENT_NAME
)

summary_row = {

    "Experiment":
    EXPERIMENT_NAME,

    "Frames":
    overall.loc[
        EXPERIMENT_NAME,
        "num_frames"
    ],

    "MOTA":
    overall.loc[
        EXPERIMENT_NAME,
        "mota"
    ],

    "MOTP":
    overall.loc[
        EXPERIMENT_NAME,
        "motp"
    ],

    "IDF1":
    overall.loc[
        EXPERIMENT_NAME,
        "idf1"
    ],

    "Precision":
    overall.loc[
        EXPERIMENT_NAME,
        "precision"
    ],

    "Recall":
    overall.loc[
        EXPERIMENT_NAME,
        "recall"
    ],

    "IDSW":
    overall.loc[
        EXPERIMENT_NAME,
        "num_switches"
    ],

    "FP":
    overall.loc[
        EXPERIMENT_NAME,
        "num_false_positives"
    ],

    "FN":
    overall.loc[
        EXPERIMENT_NAME,
        "num_misses"
    ]
}

summary_df = pd.DataFrame(
    [summary_row]
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

# =====================================================
# PRINT
# =====================================================

print("\n")
print("=" * 70)
print("PER SEQUENCE RESULTS")
print("=" * 70)
print(per_seq_df)

print("\n")
print("=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)
print(summary_df)

print("\nSaved:")
print(PER_SEQUENCE_FILE)
print(SUMMARY_FILE)