import os
import cv2
import time
from collections import defaultdict
from ultralytics import YOLO

# =====================================================
# CONFIG
# =====================================================
import torch

print("CUDA:", torch.cuda.is_available())
print("DEVICE:", torch.cuda.get_device_name(0))
BASE_DIR = "../data/VisDrone2019-MOT-val"

SEQUENCES_DIR = os.path.join(
    BASE_DIR,
    "sequences"
)

# EXPERIMENT_NAME="E03_defaultconf_person_ft"
EXPERIMENT_NAME = "E05_bottrack_person_base"

# MODEL_PATH="/userhome/phd/gayathri.rangu/AerialGuardianMOT/scripts/runs/runs/visdrone/E03_person_ft-3/weights/best.pt"
OUTPUT_VIDEO_DIR = os.path.join(
    "../outputs",
    EXPERIMENT_NAME,
    "videos"
)

OUTPUT_TRACK_DIR = os.path.join(
    "../outputs",
    EXPERIMENT_NAME,
    "tracker_outputs"
)

os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_TRACK_DIR, exist_ok=True)

MODEL_PATH = "../yolo26n.pt"

PERSON_CLASS = 0
MAX_TRAIL = 30

# =====================================================
# MODEL
# =====================================================

model = YOLO(MODEL_PATH)

# =====================================================
# ALL SEQUENCES
# =====================================================

all_sequences = sorted([
    x for x in os.listdir(SEQUENCES_DIR)
    if os.path.isdir(
        os.path.join(SEQUENCES_DIR, x)
    )
])

grand_total_frames = 0
grand_total_time = 0.0

# =====================================================
# LOOP OVER ALL VIDEOS
# =====================================================

for seq_name in all_sequences:

    print("\n")
    print("=" * 60)
    print("Processing:", seq_name)
    print("=" * 60)

    seq_dir = os.path.join(
        SEQUENCES_DIR,
        seq_name
    )

    frames = sorted(os.listdir(seq_dir))

    if len(frames) == 0:
        continue

    first_frame = cv2.imread(
        os.path.join(seq_dir, frames[0])
    )

    h, w = first_frame.shape[:2]

    output_video = os.path.join(
        OUTPUT_VIDEO_DIR,
        f"{seq_name}.mp4"
    )

    output_txt = os.path.join(
        OUTPUT_TRACK_DIR,
        f"{seq_name}.txt"
    )

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (w, h)
    )

    track_history = defaultdict(list)

    total_frames = 0
    total_time = 0.0
    total_tracks = 0

    mot_lines = []

    # =================================================
    # FRAME LOOP
    # =================================================

    for frame_idx, frame_name in enumerate(frames, start=1):

        frame_path = os.path.join(
            seq_dir,
            frame_name
        )

        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        start = time.perf_counter()

        # results = model.track(
        #     frame,
        #     imgsz=1280,
        #     persist=True,
        #     tracker="bytetrack.yaml",
        #     classes=[PERSON_CLASS],
        #     verbose=False
        # )[0] #E02 uncomment
        results = model.track(
            frame,
            imgsz=1280,
            persist=True,
            device=0,
            tracker="botsort.yaml",
            classes=[PERSON_CLASS],
            verbose=False
        )[0] #E01 uncomment
    #     results = model.track(
    #     frame,
    #     imgsz=1280,
    #     device=0,
    #     persist=True,
    #     conf=0.40,
    #     tracker="/userhome/phd/gayathri.rangu/AerialGuardianMOT/scripts/bytetrack_gmc.yaml",
    #     classes=[PERSON_CLASS],
    #     verbose=False
    # )[0] #E04CONF40
    #     results = model.track(
    #     frame,
    #     imgsz=1280,
    #     device=0,
    #     persist=True,
    #     tracker="/userhome/phd/gayathri.rangu/AerialGuardianMOT/scripts/bytetrack_gmc.yaml",
    #     classes=[PERSON_CLASS],
    #     verbose=False
    # )[0]
        end = time.perf_counter()

        frame_time = end - start

        total_time += frame_time
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

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2
                )

                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                history = track_history[track_id]

                history.append((cx, cy))

                if len(history) > MAX_TRAIL:
                    history.pop(0)

                for j in range(1, len(history)):

                    cv2.line(
                        frame,
                        history[j-1],
                        history[j],
                        (0,0,255),
                        2
                    )

        writer.write(frame)

    # =================================================
    # SAVE TRACKS
    # =================================================

    with open(output_txt, "w") as f:
        f.writelines(mot_lines)

    writer.release()

    avg_time = total_time / total_frames
    fps = 1.0 / avg_time

    print(f"Frames      : {total_frames}")
    print(f"FPS         : {fps:.2f}")
    print(f"Tracks/frame: {total_tracks/total_frames:.2f}")

    grand_total_frames += total_frames
    grand_total_time += total_time

# =====================================================
# OVERALL EXPERIMENT SUMMARY
# =====================================================

overall_fps = grand_total_frames / grand_total_time

print("\n")
print("=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)
print("Experiment :", EXPERIMENT_NAME)
print("Sequences  :", len(all_sequences))
print("Frames     :", grand_total_frames)
print("Overall FPS:", round(overall_fps, 2))
print("=" * 70)