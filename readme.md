# MirrorForm: MR-based Personal Trainer Prototype

MirrorForm is a prototype exercise analysis system that estimates human pose from a side-view RGB video and evaluates squat form.

This project was developed as a prototype for the Mixed Reality Systems course report.

---

## Features

- Pose estimation using **MediaPipe Pose**
- Automatic squat repetition counting
- Knee angle estimation
- Hip angle estimation
- Torso lean estimation
- Squat depth estimation
- Upper-back posture estimation
- Foot balance estimation
- Automatic warning generation
- Per-repetition evaluation report

---

## Directory Structure

```text
.
├── figures/            # Figures used in the report
├── models/             # MediaPipe model
├── outputs/            # Generated videos, CSVs, and plots
├── scripts/
│   ├── analyze_squat.py
│   ├── visualize_squat_results.py
│   └── make_result_overview.py
├── videos/             # Input videos
├── report/             # LaTeX source
└── README.md
```

---

## Requirements

- Python 3.11+
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Matplotlib
- Pillow
- PyMuPDF

Install the required packages:

```bash
pip install \
    mediapipe \
    opencv-python \
    numpy \
    pandas \
    matplotlib \
    pillow \
    pymupdf
```

---

## Usage

### 1. Analyze a squat video

```bash
python scripts/analyze_squat.py \
    --input videos/squat.mp4 \
    --output outputs/squat_result.mp4 \
    --csv outputs/squat_analysis.csv
```

Outputs

- Annotated video
- Frame-by-frame CSV

---

### 2. Generate evaluation plots

```bash
python scripts/visualize_squat_results.py \
    --input outputs/squat_analysis.csv \
    --timeline outputs/timeline.png \
    --rep-plot outputs/repetition_plot.png \
    --summary outputs/repetition_summary.csv
```

Outputs

- Timeline visualization
- Per-repetition evaluation
- Summary CSV

---

### 3. Create the report overview figure

```bash
python scripts/make_result_overview.py \
    --input-frame figures/input_video.png \
    --analyzed-frame figures/pose_estimation.png \
    --report-pdf figures/squat_report.pdf \
    --output-png figures/prototype_overview.png \
    --output-pdf figures/prototype_overview.pdf
```

---

## Current Prototype

The current implementation supports only a **single side-view RGB camera**.

The following functions are planned as future work.

- Multi-camera 3D pose estimation
- MR visualization using AR glasses
- Ideal pose overlay
- Voice coaching
- Personalized exercise recommendations

---

## License

This repository was created for educational purposes.
MediaPipe Pose is provided under its own license.