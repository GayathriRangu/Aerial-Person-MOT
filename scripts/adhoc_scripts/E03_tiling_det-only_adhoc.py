import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

# =====================================================
# CONFIG
# =====================================================

# SEQ_NAME = "uav0000086_00000_v"
# SEQ_NAME = "uav0000117_02622_v"
SEQ_NAME = "uav0000137_00458_v"
# SEQ_NAME = "uav0000182_00000_v"
# SEQ_NAME = "uav0000268_05773_v" #NO People or pedestrians?
# SEQ_NAME = "uav0000305_00000_v"
# SEQ_NAME = "uav0000339_00001_v"

BASE_DIR = "../data/VisDrone2019-MOT-val"

SEQ_DIR = os.path.join(
    BASE_DIR,
    "sequences",
    SEQ_NAME
)

OUTPUT_VIDEO = (
    f"../outputs/{SEQ_NAME}_tiled_bytetrack.mp4"
)

MODEL_PATH = "../yolo26n.pt"

TILE_ROWS = 2
TILE_COLS = 2

IMGSZ = 640

CONF = 0.25

# =====================================================
# MODEL
# =====================================================

model = YOLO(MODEL_PATH)

# =====================================================
# LOAD FRAMES
# =====================================================

frames = sorted(os.listdir(SEQ_DIR))

first_frame = cv2.imread(
    os.path.join(SEQ_DIR, frames[0])
)

H, W = first_frame.shape[:2]

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,
    (W, H)
)

# =====================================================
# TRAJECTORY STORAGE
# =====================================================

track_history = {}

# =====================================================
# METRICS
# =====================================================

total_frames = 0
total_time = 0
total_tracks = 0

# =====================================================
# LOOP
# =====================================================

for frame_name in frames:

    frame_path = os.path.join(
        SEQ_DIR,
        frame_name
    )

    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    start = time.time()

    all_boxes = []

    tile_h = H // TILE_ROWS
    tile_w = W // TILE_COLS

    # ==========================================
    # RUN YOLO ON EACH TILE
    # ==========================================

    for r in range(TILE_ROWS):

        for c in range(TILE_COLS):

            y1 = r * tile_h
            y2 = (r + 1) * tile_h

            x1 = c * tile_w
            x2 = (c + 1) * tile_w

            tile = frame[y1:y2, x1:x2]

            results = model.predict(
                tile,
                imgsz=IMGSZ,
                conf=CONF,
                classes=[0],   # PERSON
                verbose=False
            )[0]

            if results.boxes is None:
                continue

            for box in results.boxes:

                bx1, by1, bx2, by2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                conf = float(box.conf[0])

                all_boxes.append([
                    bx1 + x1,
                    by1 + y1,
                    bx2 + x1,
                    by2 + y1,
                    conf
                ])

    # ==========================================
    # DRAW DETECTIONS
    # ==========================================

    for det in all_boxes:

        x1, y1, x2, y2, conf = det

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0,255,0),
            2
        )

    end = time.time()

    frame_time = end - start

    total_time += frame_time
    total_frames += 1
    total_tracks += len(all_boxes)

    fps = 1 / frame_time

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    writer.write(frame)

    cv2.imshow(
        "Tiled Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =====================================================
# RESULTS
# =====================================================

writer.release()
cv2.destroyAllWindows()

avg_fps = total_frames / total_time
avg_time = total_time / total_frames

print()
print("=" * 30)
print("TILED INFERENCE RESULTS")
print("=" * 30)

print(
    f"Frames Processed : {total_frames}"
)

print(
    f"Average FPS      : {avg_fps:.2f}"
)

print(
    f"Average Time     : {avg_time*1000:.2f} ms"
)

print(
    f"Avg Dets/frame   : "
    f"{total_tracks/total_frames:.2f}"
)

print(
    f"Output Video     : {OUTPUT_VIDEO}"
)

print("=" * 30)