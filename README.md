# Aerial Person MOT

Lightweight person detection and multi-object tracking from moving drone footage using the VisDrone2019 MOT dataset.

This repository was prepared for the **Aerial Guardian** assignment. The goal is to detect and track persons from UAV imagery while balancing small-object accuracy, ID stability under drone ego-motion, model size, and runtime speed.

## Final Configuration

| Component | Choice |
|---|---|
| Detector | Fine-tuned YOLO26n |
| Tracker | BoT-SORT with global motion compensation |
| Target classes | VisDrone `pedestrian` and `person`, merged into one `person` class |
| Training image size | 1024 |
| Inference image size | 1280 |
| Confidence threshold | 0.40 |
| Hardware used | NVIDIA Tesla P100-PCIE-16GB |
| Final FPS | 40.68 FPS |
| Final checkpoint size | Approx. 5.16 MB |

The final run is:

```text
E05_bottrack_conf40_person_ft_gmc
```

## Why This Approach

Drone imagery makes people appear very small, and camera motion makes tracking harder because the whole frame shifts between consecutive images. I used a lightweight YOLO26n detector because it stays far below the 300 MB model-size budget and is suitable for later edge deployment. To adapt it to the aerial domain, I fine-tuned it on the VisDrone2019-MOT train split instead of using only generic pretrained weights.

The training split contained **24,201 images across 56 video sequences**. Its original MOT annotations are frame-level tracking labels, not YOLO detector labels, so I converted the `pedestrian` and `person` boxes into YOLO text labels using `scripts/convert_visdronemot_to_yolo.py`. This conversion was needed because YOLO training expects one label file per image with normalized center coordinates and box dimensions.

For tracking, I evaluated ByteTrack and BoT-SORT variants. The final configuration uses BoT-SORT with sparse optical-flow global motion compensation to reduce the effect of drone ego-motion during association. I also tuned the confidence threshold and tracker settings to reduce noisy detections and unstable tracks.

## Results

| Experiment | Detector | Tracker | Key setting | MOTA | IDF1 | Precision | Recall | IDSW | FPS |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| E01 | YOLO26n | ByteTrack | Baseline | 0.242 | 0.282 | 0.807 | 0.334 | 581 | 41.06 |
| E02 | Fine-tuned YOLO26n | BoT-SORT | conf=0.40 | 0.245 | 0.292 | 0.804 | 0.339 | 570 | 42.01 |
| E03 | Fine-tuned YOLO26n | ByteTrack | conf=0.40 | 0.275 | 0.369 | 0.722 | 0.469 | 684 | 42.07 |
| E04 | Fine-tuned YOLO26n | ByteTrack + GMC | tuned thresholds | 0.280 | 0.381 | 0.719 | 0.478 | 531 | 40.11 |
| E05 | Fine-tuned YOLO26n | BoT-SORT + GMC | final setting | 0.283 | 0.388 | 0.719 | 0.482 | 553 | 40.68 |

Key takeaway: fine-tuning produced the largest improvement because the generic detector missed many small aerial persons. Global motion compensation improved identity stability under drone motion. The final pipeline keeps the model lightweight while reaching approximately real-time throughput on a Tesla P100.

The full Word report includes the pipeline diagram, fine-tuning curves, PR curve, and validation prediction examples.

## Repository Layout

```text
configs/                    Dataset and tracker configuration files
docs/                       Summary report, architecture diagram, checklist
scripts/                    Conversion, training, tracking, evaluation, video scripts
results/                    Evaluation CSVs
deliverables/               Final comparison video for submission
weights/                    Lightweight final checkpoint
outputs/                    New generated videos and tracker outputs, not intended for Git
results/<experiment>/       Restored experiment videos, tracker outputs, and CSVs
data/                       Local VisDrone data, not intended for Git
Aerial_Guardian_Final_Report.docx
                            Submission report
requirements.txt            Python dependencies
Docker.dockerfile           Optional container setup
```

## Setup

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Download the VisDrone2019 MOT validation set and place it under:

```text
data/VisDrone2019-MOT-val/
  annotations/
  sequences/
```

## Run Final Tracking Pipeline

```bash
python scripts/run_inference.py ^
  --data-root data/VisDrone2019-MOT-val ^
  --model weights/aerial_guardian_yolo26n_best.pt ^
  --tracker configs/bytetrack_gmc.yaml ^
  --experiment laptop_inference_final ^
  --imgsz 1280 ^
  --conf 0.40
```

Generated outputs:

```text
outputs/laptop_inference_final/videos/
outputs/laptop_inference_final/tracker_outputs/
```

The videos contain bounding boxes, track IDs, and short trajectory tails. The tracker outputs are written in MOT-style text format.

## Submission Video

YouTube link: [Aerial Person Detection and Tracking from Drone Footage](https://youtu.be/mpWsv7NLktM)

The following video is the full validation compilation:

```text
deliverables/baseline_vs_final_all_validation_youtube.mp4
```

This is a side-by-side comparison of the baseline pipeline and the final proposed pipeline across all validation videos, with sequence cards, metric context, intro credentials, and a watermark. It is intentionally much smaller than committing the full generated `outputs/` directory.

To regenerate it from restored experiment videos:

```bash
python scripts/make_comparison_video.py --compile-all
```

## Train / Fine-Tune

Convert VisDrone MOT annotations into YOLO-style labels:

```bash
python scripts/convert_visdronemot_to_yolo.py
```

Fine-tune the detector:

```bash
python scripts/train_visdrone_person.py
```

The fine-tuning setup maps VisDrone `pedestrian` and `person` categories into one `person` class and trains at image size 1024 with augmentation for small-object robustness. The training dataset is not committed to this repository because of its size; it was prepared in a separate working dataset folder.

## Evaluate

Run MOT evaluation:

```bash
python scripts/evaluate_mot.py
```

Metrics reported include MOTA, MOTP, IDF1, precision, recall, ID switches, false positives, and false negatives. Evaluation keeps only the VisDrone `pedestrian` and `person` categories.

## Edge Deployment Notes

For deployment on NVIDIA Jetson, Qualcomm RB5 Robotics Board, or similar edge hardware:

- export the detector to ONNX or TensorRT,
- use FP16 inference where supported,
- keep batch size at 1 for streaming drone input,
- tune input size between 960 and 1280 depending on speed requirements,
- keep ReID disabled unless the target scene requires stronger occlusion recovery,
- optionally use frame skipping or adaptive resolution for real-time flight constraints.

## Known Limitation

Trajectory tails are currently drawn in raw image coordinates. When the drone camera moves sharply, the visible tail may shift even if the person motion is smooth. The next improvement would be to transform historical tail points using estimated camera motion, or render tails in a stabilized coordinate frame.
