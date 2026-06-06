import os
import cv2
import time
from collections import defaultdict
from ultralytics import YOLO

# =====================================================
# CONFIG
# =====================================================

SEQ_NAME = "uav0000086_00000_v"

BASE_DIR = "../data/VisDrone2019-MOT-val"

SEQ_DIR = os.path.join(
    BASE_DIR,
    "sequences",
    SEQ_NAME
)

OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRACKER_OUTPUT_DIR = "../tracker_outputs"
os.makedirs(TRACKER_OUTPUT_DIR, exist_ok=True)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    f"{SEQ_NAME}_bytetrack.mp4"
)

OUTPUT_TXT = os.path.join(
    TRACKER_OUTPUT_DIR,
    f"{SEQ_NAME}.txt"
)

PERSON_CLASS = 0
MAX_TRAIL = 30

# =====================================================
# MODEL
# =====================================================

model = YOLO("../yolo26n.pt")

# =====================================================
# LOAD FRAMES
# =====================================================

frames = sorted(os.listdir(SEQ_DIR))

if len(frames) == 0:
    raise RuntimeError("No frames found")

# =====================================================
# VIDEO WRITER
# =====================================================

first_frame = cv2.imread(
    os.path.join(SEQ_DIR, frames[0])
)

h, w = first_frame.shape[:2]

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (w, h)
)

# =====================================================
# TRACK HISTORY
# =====================================================

track_history = defaultdict(list)

# =====================================================
# METRICS
# =====================================================

total_frames = 0
total_time = 0.0
total_tracks = 0

# =====================================================
# SAVE MOT RESULTS
# =====================================================

mot_lines = []

# =====================================================
# MAIN LOOP
# =====================================================

for frame_idx, frame_name in enumerate(frames, start=1):

    frame_path = os.path.join(
        SEQ_DIR,
        frame_name
    )

    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    start = time.perf_counter()

    results = model.track(
        frame,
        imgsz=1280,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[PERSON_CLASS],
        verbose=False
    )[0]

    end = time.perf_counter()

    inference_time = end - start

    total_time += inference_time
    total_frames += 1

    if results.boxes.id is not None:

        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        total_tracks += len(ids)

        for box, track_id, conf in zip(
            boxes,
            ids,
            confs
        ):

            x1, y1, x2, y2 = map(int, box)

            width = x2 - x1
            height = y2 - y1

            # ==========================================
            # SAVE TRACKER OUTPUT
            # MOT FORMAT:
            # frame,id,left,top,width,height,
            # score,-1,-1,-1
            # ==========================================

            mot_lines.append(
                f"{frame_idx},"
                f"{track_id},"
                f"{x1},"
                f"{y1},"
                f"{width},"
                f"{height},"
                f"{conf:.4f},"
                f"-1,-1,-1\n"
            )

            # ==========================================
            # DRAW BOX
            # ==========================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = (
                f"ID {track_id} "
                f"{conf:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # ==========================================
            # TRAJECTORY
            # ==========================================

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            history = track_history[track_id]

            history.append((cx, cy))

            if len(history) > MAX_TRAIL:
                history.pop(0)

            for j in range(1, len(history)):

                cv2.line(
                    frame,
                    history[j - 1],
                    history[j],
                    (0, 0, 255),
                    2
                )

    writer.write(frame)

    cv2.imshow(
        "YOLO + ByteTrack",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

# =====================================================
# SAVE TXT FILE
# =====================================================

with open(OUTPUT_TXT, "w") as f:
    f.writelines(mot_lines)

# =====================================================
# CLEANUP
# =====================================================

writer.release()
cv2.destroyAllWindows()

# =====================================================
# SUMMARY
# =====================================================

avg_time = total_time / total_frames
fps = 1.0 / avg_time

avg_tracks = total_tracks / total_frames

print("\n==============================")
print("BASELINE RESULTS")
print("==============================")
print(f"Frames Processed : {total_frames}")
print(f"Average FPS      : {fps:.2f}")
print(f"Average Time     : {avg_time*1000:.2f} ms")
print(f"Avg Tracks/frame : {avg_tracks:.2f}")
print(f"Output Video     : {OUTPUT_VIDEO}")
print(f"Tracker Output   : {OUTPUT_TXT}")
print("==============================")