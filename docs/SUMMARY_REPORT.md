# Summary Report: The Aerial Guardian

This is the Markdown companion to the Word report. The final submission report is available as:

```text
Aerial_Guardian_Final_Report.docx
```

## Final Pipeline

| Component | Choice |
|---|---|
| Detector | Fine-tuned YOLO26n |
| Tracker | BoT-SORT with sparse optical-flow global motion compensation |
| Target classes | VisDrone pedestrian and person merged into one person class |
| Training image size | 1024 |
| Inference image size | 1280 |
| Confidence threshold | 0.40 |
| Hardware | NVIDIA Tesla P100-PCIE-16GB |
| FPS | 40.68 |
| Checkpoint size | Approx. 5.16 MB |

## Training Data Preparation

Fine-tuning was performed on the VisDrone2019-MOT train split, which contains 24,201 images across 56 video sequences. The train split is not stored in this lightweight submission repository, but it was prepared in a separate working dataset folder.

VisDrone MOT annotations are designed for tracking evaluation and contain frame ID, object ID, bounding box, score, class, truncation, and occlusion fields. YOLO detector training requires one text label file per image, with normalized box center coordinates and width/height. Therefore, `scripts/convert_visdronemot_to_yolo.py` was used to convert the original MOT annotations into YOLO format and to merge VisDrone `pedestrian` and `person` categories into one `person` class.

The locally available validation annotations contain 50,312 merged person/pedestrian boxes. The median annotated person size is approximately 33 x 66 pixels, which is why higher-resolution training/inference was prioritized.

## Rationale

YOLO26n was selected because it is lightweight and suitable for edge deployment while staying far below the 300 MB assignment limit. The detector was fine-tuned on VisDrone person/pedestrian annotations because generic pretrained weights miss many small aerial persons.

BoT-SORT with global motion compensation was used for the final tracking configuration. Sparse optical-flow GMC helps account for drone ego-motion before association, reducing unstable identity matching when the whole frame shifts.

## Fine-Tuning Performance

The detector fine-tuning run completed 50 epochs with image size 1024, batch size 16, and AdamW optimization. By the final epoch, the detector reached approximately:

| Metric | Value |
|---|---:|
| Precision | 0.901 |
| Recall | 0.840 |
| mAP50 | 0.912 |
| mAP50-95 | 0.546 |

The fine-tuning results and visual diagnostics are stored under:

```text
results/finetuning/person-finetunedYOLO26n/
```

Key figures include `results.png` for training curves, `BoxPR_curve.png` for the precision-recall curve, `confusion_matrix_normalized.png` for class-level behavior, and `val_batch0_pred.jpg` for qualitative validation predictions. These results support the choice to adapt a lightweight detector to the aerial-person domain instead of using a generic pretrained model directly.

## Results

| Experiment | Detector | Tracker | Key setting | MOTA | IDF1 | Precision | Recall | IDSW | FPS |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| E01 | YOLO26n | ByteTrack | Baseline | 0.242 | 0.282 | 0.807 | 0.334 | 581 | 41.06 |
| E02 | Fine-tuned YOLO26n | BoT-SORT | conf=0.40 | 0.245 | 0.292 | 0.804 | 0.339 | 570 | 42.01 |
| E03 | Fine-tuned YOLO26n | ByteTrack | conf=0.40 | 0.275 | 0.369 | 0.722 | 0.469 | 684 | 42.07 |
| E04 | Fine-tuned YOLO26n | ByteTrack + GMC | tuned thresholds | 0.280 | 0.381 | 0.719 | 0.478 | 531 | 40.11 |
| E05 | Fine-tuned YOLO26n | BoT-SORT + GMC | final setting | 0.283 | 0.388 | 0.719 | 0.482 | 553 | 40.68 |

E05 is the final selected configuration because it gives the best MOTA and IDF1 while staying near real-time speed. E04 is useful as an ablation because it produced fewer ID switches, but E05 gives the best balanced result across the reported metrics.

Final output videos are generated under:

```text
results/E05_bottrack_conf40_person_ft_gmc/videos/
```

The primary submission comparison video is:

```text
https://youtu.be/mpWsv7NLktM
```

Local copy:

```text
deliverables/baseline_vs_final_all_validation_youtube.mp4
```

## Known Limitation

Trajectory tails are drawn in raw image coordinates. During sharp drone motion, the tail can appear to move abruptly because historical points are not camera-motion compensated. The next improvement is to transform tail history using the estimated global camera motion or render tails in a stabilized coordinate frame.

## Edge Deployment Plan

- Export YOLO26n to ONNX or TensorRT.
- Use FP16 inference on NVIDIA Jetson or Qualcomm RB5 Robotics Board.
- Keep batch size at 1 for streaming drone video.
- Tune input size between 960 and 1280 depending on speed and altitude.
- Keep ReID disabled unless occlusion-heavy scenes require it.
- Use adaptive resolution or frame skipping for stricter real-time constraints.
