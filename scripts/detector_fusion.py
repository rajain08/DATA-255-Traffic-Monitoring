
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion
from sklearn.metrics import accuracy_score, balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
SAVE_DIR = OUTPUTS / "fusion"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["baseline", "p2", "frcnn"]
CLASSES = ["car", "bus", "van", "others"]
BOX_COLS = ["x1", "y1", "x2", "y2"]

COLUMNS = [
    "image", "sequence", "frame", "weather",
    "image_width", "image_height", "class_id",
    "class_name", "confidence", *BOX_COLS
]

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)

# Weight order: baseline YOLO, YOLO + P2, Faster R-CNN
CONFIGS = {
    "equal": [1.0, 1.0, 1.0],
    "baseline_heavy": [2.0, 1.0, 1.0],
    "p2_heavy": [1.0, 2.0, 1.0],
    "frcnn_heavy": [1.0, 1.0, 2.0],
    "yolo_heavy": [2.0, 2.0, 1.0],
    "fast_yolo": [1.0, 2.0, 0.0]
}

IOU_THRESHOLD = 0.55

PREFILTER = {
    "baseline": 0.05,
    "p2": 0.05,
    "frcnn": 0.10
}


def prediction_paths(split):

    return {
        "baseline": OUTPUTS / "yolov10_baseline"
                    / f"yolo_detections_{split}.csv",
        "p2": OUTPUTS / "yolov10_p2"
              / f"p2_detections_{split}.csv",
        "frcnn": OUTPUTS / "faster_rcnn"
                 / f"frcnn_detections_{split}.csv"
    }


def load_manifest(split):

    df = pd.read_csv(DATA / "frame_manifest.csv")

    df = df[df["split"] == split].copy()

    df["image"] = df["image_path"].map(
        lambda path: Path(path).stem
    )

    assert df["image"].is_unique

    return df.reset_index(drop=True)


def load_ground_truth(split):

    return pd.read_csv(
        OUTPUTS / "yolov10_baseline"
        / f"ground_truth_{split}.csv"
    )


def load_predictions(split, manifest):

    predictions = {}

    images = set(manifest["image"])

    for name, path in prediction_paths(split).items():

        df = pd.read_csv(path)

        missing = set(COLUMNS) - set(df.columns)

        if missing:
            raise ValueError(
                f"{name}: missing columns {missing}"
            )

        assert set(df["image"]) <= images
        assert df["class_id"].isin(range(4)).all()

        # Remove invalid boxes and confidence scores.
        df = df[
            (df["x2"] > df["x1"]) &
            (df["y2"] > df["y1"]) &
            df["confidence"].between(0, 1)
        ].copy()

        predictions[name] = df

        print(f"{name}: {len(df):,} detections")

    return predictions


def get_brightness(manifest, split):

    rows = []

    print(f"Calculating brightness for {split} images")

    for row in manifest.itertuples(index=False):

        path = (
            DATA / "images" / split /
            Path(row.image_path).name
        )

        if not path.exists():
            raise FileNotFoundError(path)

        with Image.open(path) as image:

            brightness = float(
                np.asarray(image.convert("L")).mean()
            )

        rows.append({
            "image": row.image,
            "weather": row.weather,
            "brightness": brightness
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        SAVE_DIR / f"brightness_{split}.csv",
        index=False
    )

    return result


# Choose the day/night threshold using validation only.
def select_brightness_threshold(validation):

    actual = validation["weather"].eq("night")

    brightness = validation["brightness"]

    results = []

    for threshold in np.arange(40, 151, 0.5):

        predicted = brightness < threshold

        results.append({
            "threshold": float(threshold),
            "balanced_accuracy": balanced_accuracy_score(
                actual, predicted
            ),
            "accuracy": accuracy_score(
                actual, predicted
            )
        })

    results = pd.DataFrame(results).sort_values(
        ["balanced_accuracy", "accuracy", "threshold"],
        ascending=[False, False, True]
    )

    results.to_csv(
        SAVE_DIR / "brightness_threshold_search.csv",
        index=False
    )

    return float(results.iloc[0]["threshold"])


def assign_conditions(brightness, threshold):

    labels = np.where(
        brightness["brightness"] < threshold,
        "night",
        "day"
    )

    return dict(zip(brightness["image"], labels))


def prepare_groups(predictions):

    groups = {}

    for name, df in predictions.items():

        filtered = df[
            df["confidence"] >= PREFILTER[name]
        ]

        groups[name] = dict(
            tuple(filtered.groupby("image", sort=False))
        )

    return groups


# Weighted Boxes Fusion for one image.
def fuse_image(groups, image, weights):

    boxes_list = []
    scores_list = []
    labels_list = []
    active_weights = []

    for name, weight in zip(MODELS, weights):

        if weight <= 0:
            continue

        active_weights.append(weight)

        detections = groups[name].get(image)

        if detections is None or detections.empty:

            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])

            continue

        # A writable copy is required for newer pandas versions.
        boxes = detections[BOX_COLS].to_numpy(
            dtype=float,
            copy=True
        )

        width = detections["image_width"].iloc[0]
        height = detections["image_height"].iloc[0]

        # WBF expects normalized coordinates.
        boxes = boxes / np.array([
            width, height, width, height
        ])

        boxes = np.clip(boxes, 0, 1)

        valid = (
            (boxes[:, 2] > boxes[:, 0]) &
            (boxes[:, 3] > boxes[:, 1])
        )

        boxes_list.append(boxes[valid].tolist())

        scores_list.append(
            detections["confidence"]
            .to_numpy()[valid].tolist()
        )

        labels_list.append(
            detections["class_id"]
            .to_numpy()[valid].tolist()
        )

    if not any(len(boxes) for boxes in boxes_list):
        return []

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=active_weights,
        iou_thr=IOU_THRESHOLD,
        skip_box_thr=0.0,
        conf_type="box_and_model_avg"
    )

    return list(zip(boxes, scores, labels))


def fuse_dataset(manifest, groups, weights):

    rows = []

    start = time.perf_counter()

    total = len(manifest)

    for index, row in enumerate(
        manifest.itertuples(index=False),
        start=1
    ):

        results = fuse_image(
            groups, row.image, weights
        )

        for box, score, label in results:

            class_id = int(label)

            # Convert normalized boxes to pixel coordinates.
            x1, y1, x2, y2 = box * np.array([
                960, 540, 960, 540
            ])

            rows.append({
                "image": row.image,
                "sequence": row.sequence,
                "frame": row.frame,
                "weather": row.weather,
                "image_width": 960,
                "image_height": 540,
                "class_id": class_id,
                "class_name": CLASSES[class_id],
                "confidence": float(score),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            })

        if index % 500 == 0 or index == total:

            print(
                f"  Processed {index}/{total} images",
                flush=True
            )

    elapsed = time.perf_counter() - start

    return (
        pd.DataFrame(rows, columns=COLUMNS),
        elapsed / total
    )


def count_matrix(detections, images):

    return np.asarray([

        detections[
            detections["confidence"] >= threshold
        ]
        .groupby("image")
        .size()
        .reindex(images, fill_value=0)
        .to_numpy()

        for threshold in THRESHOLDS
    ])


def choose_count_threshold(matrix, actual):

    errors = np.abs(
        matrix - actual
    ).mean(axis=1)

    index = int(np.argmin(errors))

    return (
        float(THRESHOLDS[index]),
        float(errors[index])
    )


def combine_conditions(day, night, conditions):

    day_images = {
        image
        for image, condition in conditions.items()
        if condition == "day"
    }

    night_images = {
        image
        for image, condition in conditions.items()
        if condition == "night"
    }

    return pd.concat([
        day[day["image"].isin(day_images)],
        night[night["image"].isin(night_images)]
    ], ignore_index=True)


def main():

    print("\nLoading validation data")

    manifest = load_manifest("val")

    predictions = load_predictions(
        "val", manifest
    )

    gt = load_ground_truth("val")

    images = manifest["image"].tolist()

    assert set(gt["image"]) <= set(images)

    actual = (
        gt.groupby("image")
        .size()
        .reindex(images, fill_value=0)
        .to_numpy()
    )

    brightness = get_brightness(
        manifest, "val"
    )

    brightness_threshold = select_brightness_threshold(
        brightness
    )

    conditions = assign_conditions(
        brightness, brightness_threshold
    )

    print(
        "Brightness threshold:",
        brightness_threshold
    )

    groups = prepare_groups(predictions)

    validation_fused = {}
    matrices = {}
    static_rows = []

    # Select fusion weights on the validation set.
    for name, weights in CONFIGS.items():

        print("\nTesting:", name, weights)

        fused, seconds = fuse_dataset(
            manifest, groups, weights
        )

        matrix = count_matrix(
            fused, images
        )

        threshold, mae = choose_count_threshold(
            matrix, actual
        )

        validation_fused[name] = fused
        matrices[name] = matrix

        static_rows.append({
            "configuration": name,
            "weights": str(weights),
            "count_threshold": threshold,
            "count_mae": mae,
            "fusion_ms": seconds * 1000
        })

        print(
            "Validation count MAE:",
            round(mae, 4)
        )

        print(
            "Counting threshold:",
            threshold
        )

    static_results = pd.DataFrame(
        static_rows
    )

    static_results.to_csv(
        SAVE_DIR / "static_validation_search.csv",
        index=False
    )

    three_model_configs = [
        name
        for name, weights in CONFIGS.items()
        if all(weight > 0 for weight in weights)
    ]

    static_best = (
        static_results[
            static_results["configuration"].isin(
                three_model_configs
            )
        ]
        .sort_values(
            "count_mae", kind="stable"
        )
        .iloc[0]
    )

    static_name = str(
        static_best["configuration"]
    )

    # Evaluate separate daytime and nighttime weights.
    night_mask = np.array([
        conditions[image] == "night"
        for image in images
    ])

    condition_rows = []

    for day_name in three_model_configs:

        for night_name in three_model_configs:

            matrix = np.where(
                night_mask[None, :],
                matrices[night_name],
                matrices[day_name]
            )

            threshold, mae = choose_count_threshold(
                matrix, actual
            )

            condition_rows.append({
                "day_config": day_name,
                "night_config": night_name,
                "count_threshold": threshold,
                "count_mae": mae
            })

    condition_results = pd.DataFrame(
        condition_rows
    ).sort_values(
        "count_mae", kind="stable"
    )

    condition_results.to_csv(
        SAVE_DIR / "condition_validation_search.csv",
        index=False
    )

    condition_best = condition_results.iloc[0]

    day_name = str(
        condition_best["day_config"]
    )

    night_name = str(
        condition_best["night_config"]
    )

    fast_best = static_results[
        static_results["configuration"] == "fast_yolo"
    ].iloc[0]

    settings = {
        "brightness_threshold": brightness_threshold,
        "iou_threshold": IOU_THRESHOLD,
        "prefilter": PREFILTER,
        "static_config": static_name,
        "static_weights": CONFIGS[static_name],
        "static_threshold": float(
            static_best["count_threshold"]
        ),
        "day_config": day_name,
        "day_weights": CONFIGS[day_name],
        "night_config": night_name,
        "night_weights": CONFIGS[night_name],
        "condition_threshold": float(
            condition_best["count_threshold"]
        ),
        "fast_threshold": float(
            fast_best["count_threshold"]
        ),
        "validation_mae": {
            "fusion_static": float(
                static_best["count_mae"]
            ),
            "fusion_condition": float(
                condition_best["count_mae"]
            ),
            "fusion_fast": float(
                fast_best["count_mae"]
            )
        }
    }

    # Freeze settings before touching the test predictions.
    (SAVE_DIR / "fusion_settings.json").write_text(
        json.dumps(settings, indent=2)
    )

    validation_fused[static_name].to_csv(
        SAVE_DIR / "fusion_static_val.csv",
        index=False
    )

    combine_conditions(
        validation_fused[day_name],
        validation_fused[night_name],
        conditions
    ).to_csv(
        SAVE_DIR / "fusion_condition_val.csv",
        index=False
    )

    validation_fused["fast_yolo"].to_csv(
        SAVE_DIR / "fusion_fast_val.csv",
        index=False
    )

    print("\nFrozen settings:")
    print(json.dumps(settings, indent=2))

    # Held-out test predictions.
    print("\nLoading test data")

    test_manifest = load_manifest("test")

    test_predictions = load_predictions(
        "test", test_manifest
    )

    test_brightness = get_brightness(
        test_manifest, "test"
    )

    test_conditions = assign_conditions(
        test_brightness,
        brightness_threshold
    )

    test_brightness["predicted_condition"] = (
        test_brightness["image"].map(test_conditions)
    )

    # Actual weather is used only for evaluation.
    test_brightness["actual_condition"] = np.where(
        test_brightness["weather"] == "night",
        "night",
        "day"
    )

    test_brightness.to_csv(
        SAVE_DIR / "test_condition_predictions.csv",
        index=False
    )

    pd.DataFrame([{
        "accuracy": accuracy_score(
            test_brightness["actual_condition"],
            test_brightness["predicted_condition"]
        ),
        "balanced_accuracy": balanced_accuracy_score(
            test_brightness["actual_condition"],
            test_brightness["predicted_condition"]
        )
    }]).to_csv(
        SAVE_DIR / "condition_classification_metrics.csv",
        index=False
    )

    test_groups = prepare_groups(
        test_predictions
    )

    test_fused = {}
    timings = []

    required_configs = list(dict.fromkeys([
        static_name,
        day_name,
        night_name,
        "fast_yolo"
    ]))

    for name in required_configs:

        print("\nTest fusion:", name)

        fused, seconds = fuse_dataset(
            test_manifest,
            test_groups,
            CONFIGS[name]
        )

        test_fused[name] = fused

        timings.append({
            "configuration": name,
            "fusion_ms_per_image": seconds * 1000,
            "device": "local CPU"
        })

    test_fused[static_name].to_csv(
        SAVE_DIR / "fusion_static_test.csv",
        index=False
    )

    combine_conditions(
        test_fused[day_name],
        test_fused[night_name],
        test_conditions
    ).to_csv(
        SAVE_DIR / "fusion_condition_test.csv",
        index=False
    )

    test_fused["fast_yolo"].to_csv(
        SAVE_DIR / "fusion_fast_test.csv",
        index=False
    )

    pd.DataFrame(timings).to_csv(
        SAVE_DIR / "fusion_timing.csv",
        index=False
    )

    print("\nFusion complete.")
    print("Saved to:", SAVE_DIR)


if __name__ == "__main__":
    main()
