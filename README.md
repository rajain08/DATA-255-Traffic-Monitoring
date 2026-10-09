# UA-DETRAC Data Preparation

This folder contains only the data-preparation portion of the traffic-monitoring project. It is designed to be completed by the dataset/preprocessing team member and handed to the modeling teammates.

## Run

1. Open `notebooks/01_dataset_preparation.ipynb` in Google Colab.
2. Enable Google Drive persistence if desired.
3. Run once with `PROFILE = "smoke"`.
4. Confirm the displayed bounding boxes match the vehicles.
5. Change to `PROFILE = "course"` and rerun from the beginning.
6. Give teammates the generated `data/processed/` folder and `outputs/figures/`.

## What the notebook does

- Downloads official UA-DETRAC images and XML annotations.
- Matches video sequences with their annotations.
- Preserves the official test partition when it is available.
- Holds validation sequences out of official training data.
- Samples every fifth frame in the course profile.
- Converts `car`, `bus`, `van`, and `others` boxes to YOLO format.
- Masks UA-DETRAC ignored regions.
- Prevents any video sequence from appearing in multiple splits.
- Audits class IDs, normalized coordinates, and image-label pairing.
- Saves manifests, `data.yaml`, EDA figures, and annotation examples.

## Generated structure

```text
traffic-monitoring/
├── notebooks/
│   └── 01_dataset_preparation.ipynb
├── data/
│   ├── raw/
│   │   ├── downloads/
│   │   └── source/
│   └── processed/
│       ├── images/{train,val,test}/
│       ├── labels/{train,val,test}/
│       ├── data.yaml
│       ├── frame_manifest.csv
│       ├── annotations.csv
│       ├── sequence_manifest.csv
│       └── sequence_split.csv
├── detector_outputs/
│   ├── yolo/
│   └── faster_rcnn/
├── outputs/
│   ├── videos/
│   ├── figures/
│   └── summaries/
└── README.md
```

## Handoff checklist

- [ ] Every selected sequence appears in exactly one split.
- [ ] At least 25 converted images were checked visually.
- [ ] Annotation audit contains no errors.
- [ ] Class and weather distributions were recorded.
- [ ] Sampling stride, seed, and selected sequence list were documented.
- [ ] `data.yaml` paths were verified.
- [ ] Raw dataset files were excluded from GitHub.

Official dataset page: https://sites.google.com/view/daweidu/projects/ua-detrac

