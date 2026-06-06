import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def existing_path(candidates):
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return Path(candidates[0])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Aerial Guardian person MOT inference on VisDrone sequences."
    )
    parser.add_argument(
        "--data-root",
        default="data/VisDrone2019-MOT-val",
        help="Folder containing VisDrone sequences/ and annotations/.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="YOLO checkpoint. Defaults to weights/aerial_guardian_yolo26n_best.pt or weights/best.pt.",
    )
    parser.add_argument(
        "--tracker",
        default="configs/bytetrack_gmc.yaml",
        help="Ultralytics tracker YAML.",
    )
    parser.add_argument(
        "--experiment",
        default="laptop_inference_final",
        help="Output experiment folder name.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root folder for generated videos and tracker outputs.",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--device", default=None, help="Use 0 for CUDA GPU or cpu for CPU.")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--tail-length", type=int, default=30)
    parser.add_argument(
        "--sequence",
        default=None,
        help="Optional single sequence name, for example uav0000086_00000_v.",
    )
    return parser.parse_args()


def draw_track(frame, box, track_id, history, tail_length):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"ID {track_id}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    history.append(center)
    if len(history) > tail_length:
        history.pop(0)

    for idx in range(1, len(history)):
        cv2.line(frame, history[idx - 1], history[idx], (0, 0, 255), 2)


def main():
    args = parse_args()
    model_path = (
        Path(args.model)
        if args.model
        else existing_path(
            [
                "weights/aerial_guardian_yolo26n_best.pt",
                "weights/best.pt",
                "weights/yolo26n.pt",
            ]
        )
    )
    data_root = Path(args.data_root)
    sequences_dir = data_root / "sequences"
    output_video_dir = Path(args.output_root) / args.experiment / "videos"
    output_track_dir = Path(args.output_root) / args.experiment / "tracker_outputs"
    output_video_dir.mkdir(parents=True, exist_ok=True)
    output_track_dir.mkdir(parents=True, exist_ok=True)

    if not sequences_dir.exists():
        raise FileNotFoundError(f"Missing sequences directory: {sequences_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", model_path)
    print("Tracker:", args.tracker)
    print("Output:", Path(args.output_root) / args.experiment)

    model = YOLO(str(model_path))
    sequence_dirs = sorted(path for path in sequences_dir.iterdir() if path.is_dir())
    if args.sequence:
        sequence_dirs = [path for path in sequence_dirs if path.name == args.sequence]
        if not sequence_dirs:
            raise FileNotFoundError(f"Sequence not found: {args.sequence}")

    grand_total_frames = 0
    grand_total_time = 0.0

    for seq_dir in sequence_dirs:
        frames = sorted(seq_dir.glob("*.jpg"))
        if not frames:
            continue

        first_frame = cv2.imread(str(frames[0]))
        if first_frame is None:
            continue
        height, width = first_frame.shape[:2]

        video_path = output_video_dir / f"{seq_dir.name}.mp4"
        track_path = output_track_dir / f"{seq_dir.name}.txt"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (width, height),
        )

        track_history = defaultdict(list)
        mot_lines = []
        total_frames = 0
        total_time = 0.0
        total_tracks = 0

        print(f"\nProcessing {seq_dir.name}")

        for frame_idx, frame_path in enumerate(frames, start=1):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            start = time.perf_counter()
            results = model.track(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                persist=True,
                device=args.device,
                tracker=args.tracker,
                classes=[args.class_id],
                verbose=False,
            )[0]
            total_time += time.perf_counter() - start
            total_frames += 1

            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                ids = results.boxes.id.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                total_tracks += len(ids)

                for box, track_id, conf in zip(boxes, ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    mot_lines.append(
                        f"{frame_idx},{track_id},{x1},{y1},{x2 - x1},{y2 - y1},{conf:.4f},-1,-1,-1\n"
                    )
                    draw_track(
                        frame,
                        box,
                        track_id,
                        track_history[track_id],
                        args.tail_length,
                    )

            writer.write(frame)

        writer.release()
        track_path.write_text("".join(mot_lines), encoding="utf-8")

        if total_frames:
            fps = total_frames / total_time
            print(f"Frames: {total_frames}")
            print(f"FPS: {fps:.2f}")
            print(f"Tracks/frame: {total_tracks / total_frames:.2f}")
            grand_total_frames += total_frames
            grand_total_time += total_time

    if grand_total_frames:
        print("\n" + "=" * 70)
        print("Experiment:", args.experiment)
        print("Sequences:", len(sequence_dirs))
        print("Frames:", grand_total_frames)
        print("Overall FPS:", round(grand_total_frames / grand_total_time, 2))
        print("=" * 70)


if __name__ == "__main__":
    main()
