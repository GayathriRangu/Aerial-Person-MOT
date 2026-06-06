import os
import cv2
from collections import defaultdict

#SEQ_NAME = "uav0000086_00000_v"
# SEQ_NAME = "uav0000117_02622_v"
# SEQ_NAME = "uav0000137_00458_v"
# SEQ_NAME = "uav0000182_00000_v"
# SEQ_NAME = "uav0000268_05773_v" #NO People or pedestrians?
# SEQ_NAME = "uav0000305_00000_v"
SEQ_NAME = "uav0000339_00001_v"

BASE_DIR = "../data/VisDrone2019-MOT-val"

SEQ_DIR = os.path.join(BASE_DIR, "sequences", SEQ_NAME)
ANN_FILE = os.path.join(BASE_DIR, "annotations", SEQ_NAME + ".txt")

# frame -> list of detections
detections = defaultdict(list)

with open(ANN_FILE, "r") as f:
    for line in f:
        vals = line.strip().split(",")

        frame_id = int(vals[0])
        track_id = int(vals[1])

        x = int(vals[2])
        y = int(vals[3])
        w = int(vals[4])
        h = int(vals[5])

        cls = int(vals[7])

        # keep only pedestrians/persons
        if cls not in [1, 2]:
            continue

        detections[frame_id].append(
            (track_id, x, y, w, h)
        )

frames = sorted(os.listdir(SEQ_DIR))

for idx, frame_name in enumerate(frames[:300]):

    frame_path = os.path.join(SEQ_DIR, frame_name)

    img = cv2.imread(frame_path)

    frame_id = idx + 1

    for track_id, x, y, w, h in detections[frame_id]:

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"ID:{track_id}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    cv2.imshow("GT", img)

    key = cv2.waitKey(30)

    if key == 27:
        break

cv2.destroyAllWindows()