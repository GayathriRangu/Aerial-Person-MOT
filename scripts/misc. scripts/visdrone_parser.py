import os

class_names = {
    1: "pedestrian",
    2: "person",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor"
}

ann_file = "../data/VisDrone2019-MOT-val/annotations/uav0000086_00000_v.txt"

with open(ann_file) as f:
    for i, line in enumerate(f):
        vals = line.strip().split(",")

        frame_id = int(vals[0])
        track_id = int(vals[1])
        cls_id = int(vals[7])

        print(
            f"Frame={frame_id} "
            f"Track={track_id} "
            f"Class={class_names[cls_id]}"
        )

        if i > 20:
            break