import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


WATERMARK = "Gayathri Rangu | PhD Student, IIT Guwahati"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create baseline vs final MOT comparison videos."
    )
    parser.add_argument("--compile-all", action="store_true")
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="Baseline video directory. Auto-detects results/ then outputs/ if omitted.",
    )
    parser.add_argument(
        "--final-dir",
        default=None,
        help="Final video directory. Auto-detects results/ then outputs/ if omitted.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Single baseline video path.",
    )
    parser.add_argument(
        "--final",
        default=None,
        help="Single final video path.",
    )
    parser.add_argument(
        "--out",
        default="deliverables/baseline_vs_final_all_validation_youtube.mp4",
    )
    parser.add_argument(
        "--baseline-metrics",
        default="results/E01_Baseline_per_sequence.csv",
    )
    parser.add_argument(
        "--final-metrics",
        default="results/E05_bottrack_conf40_person_ft_gmc_per_sequence.csv",
    )
    parser.add_argument("--duration-sec", type=float, default=45.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=360)
    return parser.parse_args()


def first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return Path(paths[0])


def load_metrics(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["Sequence"]: row for row in csv.DictReader(f)}


def fmt_metric(row, key):
    if not row:
        return "n/a"
    value = row.get(key, "")
    if value in ("", "nan", "None"):
        return "n/a"
    try:
        if key in {"IDSW", "FP", "FN", "Frames"}:
            return str(int(float(value)))
        return f"{float(value):.3f}"
    except ValueError:
        return value


def add_watermark(img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (w, h), baseline = cv2.getTextSize(WATERMARK, font, scale, thickness)
    x = img.shape[1] - w - 24
    y = img.shape[0] - 22
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x - 10, y - h - 8),
        (x + w + 10, y + baseline + 8),
        (11, 37, 69),
        -1,
    )
    cv2.addWeighted(overlay, 0.42, img, 0.58, 0, img)
    cv2.putText(img, WATERMARK, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def put_label(img, text, xy, scale=0.7, color=(255, 255, 255), bg=(11, 37, 69)):
    x, y = xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 10
    cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + baseline + pad), bg, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def read_resized(cap, width, height):
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def make_intro(width, height, fps, seconds):
    frames = []
    for _ in range(int(fps * seconds)):
        canvas = np.full((height, width, 3), (248, 250, 252), dtype=np.uint8)
        cv2.putText(canvas, "Aerial Guardian MOT", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (11, 37, 69), 3, cv2.LINE_AA)
        cv2.putText(canvas, "Baseline YOLO26n + ByteTrack  vs  Fine-tuned YOLO26n + BoT-SORT + GMC", (60, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (51, 65, 85), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Final: MOTA 0.283 | IDF1 0.388 | FPS 40.68 | 5.16 MB checkpoint", (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (31, 77, 120), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Created by Gayathri Rangu", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (11, 37, 69), 2, cv2.LINE_AA)
        cv2.putText(canvas, "PhD Student, Indian Institute of Technology Guwahati", (60, 336), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (71, 85, 105), 2, cv2.LINE_AA)
        add_watermark(canvas)
        frames.append(canvas)
    return frames


def make_outro(width, height, fps, seconds):
    frames = []
    lines = [
        "Fine-tuning improves small-person detection in aerial views.",
        "Global motion compensation helps reduce drone ego-motion errors.",
        "The final model remains lightweight: ~5.16 MB checkpoint at 40.68 FPS.",
    ]
    for _ in range(int(fps * seconds)):
        canvas = np.full((height, width, 3), (248, 250, 252), dtype=np.uint8)
        cv2.putText(canvas, "Key takeaway", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (11, 37, 69), 3, cv2.LINE_AA)
        for idx, line in enumerate(lines):
            cv2.putText(canvas, line, (60, 180 + idx * 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (51, 65, 85), 2, cv2.LINE_AA)
        add_watermark(canvas)
        frames.append(canvas)
    return frames


def make_sequence_card(width, height, fps, seconds, sequence, base_row, final_row):
    frames = []
    lines = [
        f"Sequence: {sequence}",
        f"Baseline  MOTA {fmt_metric(base_row, 'MOTA')} | IDF1 {fmt_metric(base_row, 'IDF1')} | IDSW {fmt_metric(base_row, 'IDSW')}",
        f"Final     MOTA {fmt_metric(final_row, 'MOTA')} | IDF1 {fmt_metric(final_row, 'IDF1')} | IDSW {fmt_metric(final_row, 'IDSW')}",
    ]
    for _ in range(int(fps * seconds)):
        canvas = np.full((height, width, 3), (248, 250, 252), dtype=np.uint8)
        cv2.putText(canvas, "Validation Sequence Comparison", (60, 112), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (11, 37, 69), 3, cv2.LINE_AA)
        for idx, line in enumerate(lines):
            cv2.putText(canvas, line, (60, 180 + idx * 44), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (51, 65, 85), 2, cv2.LINE_AA)
        add_watermark(canvas)
        frames.append(canvas)
    return frames


def write_pair_segment(writer, baseline_path, final_path, args, dims, sequence, base_row=None, final_row=None, max_seconds=None):
    canvas_w, canvas_h, panel_w, panel_h, gap, header_h = dims
    baseline = cv2.VideoCapture(str(baseline_path))
    final = cv2.VideoCapture(str(final_path))
    if not baseline.isOpened():
        raise FileNotFoundError(f"Could not open baseline video: {baseline_path}")
    if not final.isOpened():
        raise FileNotFoundError(f"Could not open final video: {final_path}")

    source_fps = baseline.get(cv2.CAP_PROP_FPS) or args.fps
    start_frame = int(args.start_sec * source_fps)
    baseline.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    final.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_step = max(1, round(source_fps / args.fps))
    max_frames = int((max_seconds or args.duration_sec) * args.fps)
    written = 0

    while written < max_frames:
        left = read_resized(baseline, panel_w, panel_h)
        right = read_resized(final, panel_w, panel_h)
        if left is None or right is None:
            break

        for _ in range(frame_step - 1):
            baseline.grab()
            final.grab()

        canvas = np.full((canvas_h, canvas_w, 3), (245, 247, 250), dtype=np.uint8)
        cv2.putText(canvas, f"Baseline vs Final Proposed Pipeline | {sequence}", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (11, 37, 69), 2, cv2.LINE_AA)

        y0 = header_h
        canvas[y0 : y0 + panel_h, 0:panel_w] = left
        canvas[y0 : y0 + panel_h, panel_w + gap : panel_w + gap + panel_w] = right
        put_label(canvas, "Baseline: YOLO26n + ByteTrack", (18, y0 + 36), bg=(120, 60, 50))
        put_label(canvas, "Final: Fine-tuned YOLO26n + BoT-SORT + GMC", (panel_w + gap + 18, y0 + 36), bg=(31, 77, 120))
        cv2.line(canvas, (panel_w + gap // 2, y0), (panel_w + gap // 2, y0 + panel_h), (203, 213, 225), 2)

        footer_y = header_h + panel_h + 30
        metric_line = (
            f"Sequence metrics | Baseline: MOTA {fmt_metric(base_row, 'MOTA')} IDF1 {fmt_metric(base_row, 'IDF1')} "
            f"IDSW {fmt_metric(base_row, 'IDSW')} | Final: MOTA {fmt_metric(final_row, 'MOTA')} "
            f"IDF1 {fmt_metric(final_row, 'IDF1')} IDSW {fmt_metric(final_row, 'IDSW')}"
        )
        cv2.putText(canvas, metric_line, (24, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (51, 65, 85), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Shows cleaner small-person detections and camera-motion-aware track association.", (24, footer_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (71, 85, 105), 1, cv2.LINE_AA)
        add_watermark(canvas)
        writer.write(canvas)
        written += 1

    baseline.release()
    final.release()
    return written


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else first_existing(["results/E01_baseline/videos", "outputs/E01_baseline/videos"])
    final_dir = Path(args.final_dir) if args.final_dir else first_existing(["results/E05_bottrack_conf40_person_ft_gmc/videos", "outputs/E05_botsort_conf40_person_ft_gmc/videos", "outputs/E05_bottrack_conf40_person_ft_gmc/videos"])

    panel_w = args.panel_width
    panel_h = args.panel_height
    header_h = 76
    footer_h = 72
    gap = 16
    canvas_w = panel_w * 2 + gap
    canvas_h = header_h + panel_h + footer_h
    dims = (canvas_w, canvas_h, panel_w, panel_h, gap, header_h)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (canvas_w, canvas_h))
    for frame in make_intro(canvas_w, canvas_h, args.fps, 2.0):
        writer.write(frame)

    baseline_metrics = load_metrics(args.baseline_metrics)
    final_metrics = load_metrics(args.final_metrics)
    total_written = 0

    if args.compile_all:
        videos = sorted(p for p in baseline_dir.glob("*.mp4") if (final_dir / p.name).exists())
        if not videos:
            raise FileNotFoundError(f"No matching videos in {baseline_dir} and {final_dir}")
        for video in videos:
            sequence = video.stem
            base_row = baseline_metrics.get(sequence)
            final_row = final_metrics.get(sequence)
            for frame in make_sequence_card(canvas_w, canvas_h, args.fps, 1.5, sequence, base_row, final_row):
                writer.write(frame)
            total_written += write_pair_segment(writer, video, final_dir / video.name, args, dims, sequence, base_row, final_row, max_seconds=10_000)
    else:
        baseline_path = Path(args.baseline) if args.baseline else baseline_dir / "uav0000086_00000_v.mp4"
        final_path = Path(args.final) if args.final else final_dir / "uav0000086_00000_v.mp4"
        sequence = baseline_path.stem
        total_written += write_pair_segment(writer, baseline_path, final_path, args, dims, sequence, baseline_metrics.get(sequence), final_metrics.get(sequence))

    for frame in make_outro(canvas_w, canvas_h, args.fps, 2.0):
        writer.write(frame)
    writer.release()
    print(f"Wrote {out_path.resolve()} ({total_written / args.fps:.1f}s comparison content)")


if __name__ == "__main__":
    main()
