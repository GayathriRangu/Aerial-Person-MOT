# Submission Checklist

## Commit These

- `README.md`
- `requirements.txt`
- `Dockerfile`
- `configs/`
- `docs/`
- `scripts/`
- selected CSV summaries from `results/`
- `Aerial_Guardian_Final_Report.docx`
- `weights/aerial_guardian_yolo26n_best.pt`

## Do Not Commit

- `data/`
- `outputs/`
- full train/validation image folders
- all generated videos from every experiment
- `deliverables/baseline_vs_final_all_validation_youtube.mp4` because it is over GitHub's normal file limit
- large model checkpoints except the small final checkpoint in `weights/`
- environment folders such as `aerial_env/`

## Final Run To Highlight

Use this run as the final reported configuration:

```text
E05_bottrack_conf40_person_ft_gmc
```

Full generated output folder, kept local:

```text
results/E05_bottrack_conf40_person_ft_gmc/videos/
results/E05_bottrack_conf40_person_ft_gmc/tracker_outputs/
```

Submission comparison video:

```text
https://youtu.be/mpWsv7NLktM
```

Local copy, kept outside Git:

```text
deliverables/baseline_vs_final_all_validation_youtube.mp4
```

## Before Sending

1. Confirm the private GitHub/GitLab repo opens cleanly after a fresh clone.
2. Do not push the full 7 GB `outputs/` folder unless you use a separate release/artifact storage option.
3. Confirm the YouTube link is present in the README and summary report.
4. Confirm the README command uses `weights/aerial_guardian_yolo26n_best.pt`.
5. Mention the known tail limitation honestly: image-space tails move with drone camera motion, and the next refinement is camera-motion-compensated trail rendering.
