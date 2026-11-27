
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, f1_score

from detector_fusion import (
    load_manifest,
    load_ground_truth,
    prediction_paths,
    count_matrix,
    choose_count_threshold,
    PREFILTER
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FUSION_DIR = ROOT / "outputs" / "fusion"
SAVE_DIR = FUSION_DIR / "reporting"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["car", "bus", "van", "others"]
BOX_COLS = ["x1", "y1", "x2", "y2"]

LLM_MODEL = "llama3.2:3b"

UNCERTAINTY = {
    "high": "Detector agreement is relatively strong.",
    "medium": "The vehicle count is an approximate estimate.",
    "low": "The vehicle count is uncertain."
}


def get_counts(detections, images, threshold):

    return (
        detections[
            detections["confidence"] >= threshold
        ]
        .groupby("image")
        .size()
        .reindex(images, fill_value=0)
    )


def density_label(count, low_cut, high_cut):

    if count <= low_cut:
        return "low"

    if count <= high_cut:
        return "medium"

    return "high"


def box_iou(box, boxes):

    if len(boxes) == 0:
        return np.array([])

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(0, x2 - x1) *
        np.maximum(0, y2 - y1)
    )

    area1 = (
        (box[2] - box[0]) *
        (box[3] - box[1])
    )

    area2 = (
        (boxes[:, 2] - boxes[:, 0]) *
        (boxes[:, 3] - boxes[:, 1])
    )

    return intersection / np.maximum(
        area1 + area2 - intersection,
        1e-9
    )


# Calculate detector support for each fused bounding box.
def detector_agreement(fused, individual):

    if fused.empty:
        return None

    support = []

    for row in fused.itertuples(index=False):

        box = np.array([
            row.x1, row.y1, row.x2, row.y2
        ])

        votes = 0

        for detections in individual:

            overlaps = box_iou(
                box,
                detections[BOX_COLS].to_numpy(
                    dtype=float
                )
            )

            if len(overlaps) and overlaps.max() >= 0.5:
                votes += 1

        support.append(
            votes / len(individual)
        )

    return float(np.mean(support))


# Confidence tiers are heuristic rather than calibrated probabilities.
def confidence_tier(agreement, score):

    if agreement is None or score is None:
        return "low"

    if agreement >= 0.8 and score >= 0.5:
        return "high"

    if agreement >= 0.5 and score >= 0.25:
        return "medium"

    return "low"


def template_report(record):

    return (
        f"Estimated vehicles: {record['vehicle_count']}. "
        f"Scene density: {record['density']}. "
        f"Confidence tier: {record['confidence_tier']}. "
        f"Alert: {record['alert']}."
    )


# Generate structured reports using Ollama's JSON schema support.
def generate_llm_report(record):

    schema = {
        "type": "object",
        "properties": {
            "vehicle_count": {
                "type": "integer",
                "minimum": 0
            },
            "density": {
                "type": "string",
                "enum": ["low", "medium", "high"]
            },
            "confidence_tier": {
                "type": "string",
                "enum": ["low", "medium", "high"]
            },
            "alert": {
                "type": "string",
                "enum": [
                    "none",
                    "high density estimate"
                ]
            },
            "uncertainty": {
                "type": "string",
                "enum": list(
                    UNCERTAINTY.values()
                )
            }
        },
        "required": [
            "vehicle_count",
            "density",
            "confidence_tier",
            "alert",
            "uncertainty"
        ],
        "additionalProperties": False
    }

    input_data = {
        "vehicle_count": record["vehicle_count"],
        "density": record["density"],
        "confidence_tier": record["confidence_tier"],
        "alert": record["alert"]
    }

    prompt = f"""
You are a traffic monitoring reporting assistant.

Your task is to generate a structured JSON report using the input.

Copy vehicle_count, density, confidence_tier, and alert
exactly from the input.

The uncertainty field must contain one complete sentence.

Use the following mapping:

If confidence_tier is high:
Detector agreement is relatively strong.

If confidence_tier is medium:
The vehicle count is an approximate estimate.

If confidence_tier is low:
The vehicle count is uncertain.

Do not return only high, medium, or low in the uncertainty field.

Do not infer accidents, speed, movement, road closures,
or congestion causes.

Example input:

{{
    "vehicle_count": 5,
    "density": "low",
    "confidence_tier": "medium",
    "alert": "none"
}}

Example output:

{{
    "vehicle_count": 5,
    "density": "low",
    "confidence_tier": "medium",
    "alert": "none",
    "uncertainty": "The vehicle count is an approximate estimate."
}}

Return only a JSON object matching this schema:

{json.dumps(schema)}

Actual input:

{json.dumps(input_data)}
"""

    body = json.dumps({
        "model": LLM_MODEL,
        "prompt": prompt,
        "format": schema,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 300
        }
    }).encode("utf-8")

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body,
        headers={
            "Content-Type": "application/json"
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=120
    ) as response:

        result = json.loads(
            response.read()
        )

    return result["response"]


# Verify all report fields against the structured input.
def check_report(raw, record):

    expected = {
        "vehicle_count": record["vehicle_count"],
        "density": record["density"],
        "confidence_tier": record["confidence_tier"],
        "alert": record["alert"],
        "uncertainty": UNCERTAINTY[
            record["confidence_tier"]
        ]
    }

    try:

        parsed = json.loads(raw)

    except (ValueError, TypeError):

        print(
            "Verification failed: invalid JSON"
        )

        return False, None

    if not isinstance(parsed, dict):

        print(
            "Verification failed: not a JSON object"
        )

        return False, None

    if set(parsed.keys()) != set(expected.keys()):

        print(
            "Verification failed: incorrect fields"
        )

        return False, None

    if type(parsed["vehicle_count"]) is not int:

        print(
            "Verification failed: invalid count type"
        )

        return False, None

    for key, expected_value in expected.items():

        actual_value = parsed[key]

        if actual_value != expected_value:

            print(
                f"Verification failed for {key}: "
                f"expected {expected_value!r}, "
                f"received {actual_value!r}"
            )

            return False, None

    return True, parsed


def draw_boxes(image, detections, color):

    draw = ImageDraw.Draw(image)

    for row in detections.itertuples(index=False):

        draw.rectangle(
            [
                row.x1,
                row.y1,
                row.x2,
                row.y2
            ],
            outline=color,
            width=2
        )

    return image


def save_failure_case(
    path,
    name,
    predicted,
    actual
):

    with Image.open(path) as original:

        original = original.convert("RGB")

        gt_image = draw_boxes(
            original.copy(),
            actual,
            "lime"
        )

        pred_image = draw_boxes(
            original.copy(),
            predicted,
            "red"
        )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 4)
    )

    axes[0].imshow(gt_image)

    axes[0].set_title(
        f"Ground truth: {len(actual)}"
    )

    axes[1].imshow(pred_image)

    axes[1].set_title(
        f"Predicted: {len(predicted)}"
    )

    for ax in axes:

        ax.axis("off")

    plt.tight_layout()

    fig.savefig(
        SAVE_DIR / f"failure_{name}.png",
        dpi=150
    )

    plt.close(fig)


def main():

    settings = json.loads(
        (FUSION_DIR / "fusion_settings.json")
        .read_text()
    )

    val_manifest = load_manifest("val")
    test_manifest = load_manifest("test")

    val_images = val_manifest["image"].tolist()
    test_images = test_manifest["image"].tolist()

    val_gt = load_ground_truth("val")

    validation_counts = (
        val_gt.groupby("image")
        .size()
        .reindex(val_images, fill_value=0)
    )

    # Determine relative density categories from validation.
    low_cut, high_cut = np.quantile(
        validation_counts,
        [1 / 3, 2 / 3]
    )

    low_cut = int(low_cut)

    high_cut = max(
        int(high_cut),
        low_cut + 1
    )

    (SAVE_DIR / "density_thresholds.json").write_text(
        json.dumps({
            "low_max": low_cut,
            "medium_max": high_cut
        }, indent=2)
    )

    fusion_names = [
        "fusion_static",
        "fusion_condition",
        "fusion_fast"
    ]

    val_paths = prediction_paths("val")

    val_paths.update({
        name: FUSION_DIR / f"{name}_val.csv"
        for name in fusion_names
    })

    test_paths = prediction_paths("test")

    test_paths.update({
        name: FUSION_DIR / f"{name}_test.csv"
        for name in fusion_names
    })

    thresholds = {}
    validation_mae = {}

    # Select each counting threshold using validation only.
    for name, path in val_paths.items():

        detections = pd.read_csv(path)

        matrix = count_matrix(
            detections,
            val_images
        )

        threshold, mae = choose_count_threshold(
            matrix,
            validation_counts.to_numpy()
        )

        thresholds[name] = threshold
        validation_mae[name] = mae

    # Choose the strongest fused system on validation.
    selected = min(
        fusion_names,
        key=lambda name: validation_mae[name]
    )

    print(
        "Selected fusion system:",
        selected
    )

    print(
        "Density thresholds:",
        low_cut,
        high_cut
    )

    (SAVE_DIR / "reporting_settings.json").write_text(
        json.dumps({
            "selected_system": selected,
            "thresholds": thresholds,
            "validation_mae": validation_mae,
            "llm_model": LLM_MODEL,
            "llm_format": "json_schema",
            "llm_temperature": 0,
            "llm_seed": 42
        }, indent=2)
    )

    test_gt = load_ground_truth("test")

    actual = (
        test_gt.groupby("image")
        .size()
        .reindex(test_images, fill_value=0)
    )

    actual_density = actual.map(
        lambda count: density_label(
            count,
            low_cut,
            high_cut
        )
    )

    predictions = {
        name: pd.read_csv(path)
        for name, path in test_paths.items()
    }

    # Compare counting and density classification.
    density_results = []

    for name, detections in predictions.items():

        estimated = get_counts(
            detections,
            test_images,
            thresholds[name]
        )

        estimated_density = estimated.map(
            lambda count: density_label(
                count,
                low_cut,
                high_cut
            )
        )

        density_results.append({
            "system": name,
            "threshold": thresholds[name],
            "count_mae": float(
                np.abs(estimated - actual).mean()
            ),
            "density_accuracy": accuracy_score(
                actual_density,
                estimated_density
            ),
            "density_macro_f1": f1_score(
                actual_density,
                estimated_density,
                labels=[
                    "low",
                    "medium",
                    "high"
                ],
                average="macro",
                zero_division=0
            )
        })

    density_table = pd.DataFrame(
        density_results
    )

    density_table.to_csv(
        SAVE_DIR / "density_metrics.csv",
        index=False
    )

    print("\nDensity and count evaluation:")

    print(
        density_table.to_string(index=False)
    )

    chosen = predictions[selected]

    chosen = chosen[
        chosen["confidence"] >= thresholds[selected]
    ].copy()

    selected_groups = dict(
        tuple(chosen.groupby("image"))
    )

    individual_names = (
        ["baseline", "p2"]
        if selected == "fusion_fast"
        else ["baseline", "p2", "frcnn"]
    )

    individual_groups = {}

    for name in individual_names:

        df = predictions[name]

        df = df[
            df["confidence"] >= PREFILTER[name]
        ]

        individual_groups[name] = dict(
            tuple(df.groupby("image"))
        )

    condition_map = pd.read_csv(
        FUSION_DIR / "test_condition_predictions.csv"
    ).set_index(
        "image"
    )["predicted_condition"].to_dict()

    print("\nBuilding structured traffic records")

    records = []

    for image in test_images:

        boxes = selected_groups.get(
            image,
            chosen.iloc[:0]
        )

        individual = [
            individual_groups[name].get(
                image,
                predictions[name].iloc[:0]
            )
            for name in individual_names
        ]

        agreement = detector_agreement(
            boxes,
            individual
        )

        mean_score = (
            float(boxes["confidence"].mean())
            if len(boxes)
            else None
        )

        count = len(boxes)

        density = density_label(
            count,
            low_cut,
            high_cut
        )

        tier = confidence_tier(
            agreement,
            mean_score
        )

        records.append({
            "image": image,
            "system": selected,
            "condition": condition_map[image],
            "vehicle_count": count,
            "class_counts": {
                name: int(
                    (boxes["class_name"] == name).sum()
                )
                for name in CLASSES
            },
            "density": density,
            "mean_confidence": mean_score,
            "detector_agreement": agreement,
            "confidence_tier": tier,
            "alert": (
                "high density estimate"
                if density == "high"
                else "none"
            )
        })

    (SAVE_DIR / "traffic_records.json").write_text(
        json.dumps(
            records,
            indent=2,
            allow_nan=False
        )
    )

    records_df = pd.DataFrame(records)

    records_df["ground_truth_count"] = (
        records_df["image"].map(actual)
    )

    records_df["absolute_error"] = (
        records_df["vehicle_count"]
        - records_df["ground_truth_count"]
    ).abs()

    records_df.to_csv(
        SAVE_DIR / "traffic_records.csv",
        index=False
    )

    # Analyze confidence tiers against actual counting errors.
    confidence_summary = (
        records_df.groupby("confidence_tier")
        .agg(
            frames=("image", "size"),
            count_mae=("absolute_error", "mean"),
            mean_agreement=("detector_agreement", "mean")
        )
        .reset_index()
    )

    confidence_summary.to_csv(
        SAVE_DIR / "confidence_analysis.csv",
        index=False
    )

    print("\nConfidence analysis:")

    print(
        confidence_summary.to_string(index=False)
    )

    # Use the same reproducible sample as the original experiment.
    sample = pd.concat([
        group.sample(
            n=min(5, len(group)),
            random_state=42
        )
        for _, group in records_df.groupby([
            "condition",
            "confidence_tier"
        ])
    ])

    lookup = {
        record["image"]: record
        for record in records
    }

    report_rows = []

    print("\nGenerating LLM reports")

    for image in sample["image"]:

        record = lookup[image]

        fixed = template_report(record)

        try:

            raw = generate_llm_report(record)

            valid, parsed = check_report(
                raw,
                record
            )

            status = "completed"

        except Exception as error:

            print("LLM error:", error)

            raw = ""
            valid = None
            parsed = None
            status = "unavailable"

        if valid:

            final = (
                f"An estimated {parsed['vehicle_count']} "
                f"vehicles are visible. "
                f"Scene density is {parsed['density']}. "
                f"Confidence tier: "
                f"{parsed['confidence_tier']}. "
                f"{parsed['uncertainty']} "
                f"Alert: {parsed['alert']}."
            )

        else:

            # Fall back to a deterministic report if verification fails.
            final = fixed

        report_rows.append({
            "image": image,
            "condition": record["condition"],
            "confidence_tier": record["confidence_tier"],
            "vehicle_count": record["vehicle_count"],
            "ground_truth_count": int(actual[image]),
            "density": record["density"],
            "ground_truth_density": actual_density[image],
            "template": fixed,
            "llm_raw": raw,
            "llm_supported": valid,
            "status": status,
            "final_report": final
        })

        print(
            image,
            status,
            valid,
            flush=True
        )

    reports = pd.DataFrame(report_rows)

    reports.to_csv(
        SAVE_DIR / "report_evaluation.csv",
        index=False
    )

    completed = reports[
        reports["status"] == "completed"
    ]

    verified_count = int(
        completed["llm_supported"].eq(True).sum()
    )

    invalid_count = len(completed) - verified_count

    report_metrics = pd.DataFrame([{
        "llm_model": LLM_MODEL,
        "prompting_method": "json_schema",
        "attempted_reports": len(reports),
        "completed_reports": len(completed),
        "verified_reports": verified_count,
        "invalid_reports": invalid_count,
        "unavailable_reports": (
            len(reports) - len(completed)
        ),
        "verification_pass_rate": (
            verified_count / len(completed)
            if len(completed)
            else np.nan
        ),
        "unsupported_or_invalid_report_rate": (
            invalid_count / len(completed)
            if len(completed)
            else np.nan
        ),
        "fallback_reports": (
            len(reports) - verified_count
        ),
        "template_input_consistency": 1.0,
        "sample_count_mae": float(
            np.abs(
                reports["vehicle_count"]
                - reports["ground_truth_count"]
            ).mean()
        ),
        "sample_density_accuracy": accuracy_score(
            reports["ground_truth_density"],
            reports["density"]
        )
    }])

    report_metrics.to_csv(
        SAVE_DIR / "report_metrics.csv",
        index=False
    )

    print("\nLLM report evaluation:")

    print(
        report_metrics.to_string(index=False)
    )

    # Compare original prompting against schema-constrained generation.
    original_path = (
        SAVE_DIR / "report_evaluation_original.csv"
    )

    if original_path.exists():

        original = pd.read_csv(
            original_path
        )

        original_completed = original[
            original["status"] == "completed"
        ]

        original_verified = int(
            original_completed["llm_supported"]
            .astype(str)
            .str.lower()
            .eq("true")
            .sum()
        )

        original_invalid = (
            len(original_completed)
            - original_verified
        )

        same_sample = (
            sorted(original["image"].tolist())
            == sorted(reports["image"].tolist())
        )

        comparison = pd.DataFrame([
            {
                "method": "original_json_prompt",
                "attempted": len(original),
                "completed": len(original_completed),
                "verified": original_verified,
                "invalid": original_invalid,
                "pass_rate": (
                    original_verified / len(original_completed)
                    if len(original_completed)
                    else np.nan
                ),
                "same_sample": same_sample
            },
            {
                "method": "schema_constrained",
                "attempted": len(reports),
                "completed": len(completed),
                "verified": verified_count,
                "invalid": invalid_count,
                "pass_rate": (
                    verified_count / len(completed)
                    if len(completed)
                    else np.nan
                ),
                "same_sample": same_sample
            }
        ])

        comparison.to_csv(
            SAVE_DIR / "llm_prompt_comparison.csv",
            index=False
        )

        print("\nOriginal vs schema-constrained reporting:")

        print(
            comparison.to_string(index=False)
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.bar(
            ["Original JSON", "JSON Schema"],
            comparison["pass_rate"] * 100,
            color=["gray", "steelblue"]
        )

        ax.set_ylabel(
            "Verification Pass Rate (%)"
        )

        ax.set_ylim(0, 110)

        ax.set_title(
            "LLM Report Verification Comparison"
        )

        for index, value in enumerate(
            comparison["pass_rate"] * 100
        ):

            if pd.notna(value):

                ax.text(
                    index,
                    value + 2,
                    f"{value:.1f}%",
                    ha="center"
                )

        ax.grid(
            axis="y",
            alpha=0.3
        )

        plt.tight_layout()

        fig.savefig(
            SAVE_DIR / "llm_prompt_comparison.png",
            dpi=150
        )

        plt.close(fig)

    image_paths = dict(zip(
        test_manifest["image"],
        test_manifest["image_path"].map(
            lambda path: (
                DATA / "images" / "test"
                / Path(path).name
            )
        )
    ))

    # Visualize the four largest counting errors.
    print("\nCreating failure-case examples")

    for row in records_df.nlargest(
        4,
        "absolute_error"
    ).itertuples():

        save_failure_case(
            image_paths[row.image],
            row.image,
            selected_groups.get(
                row.image,
                chosen.iloc[:0]
            ),
            test_gt[
                test_gt["image"] == row.image
            ]
        )

    # Compare density classification across systems.
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    density_table.plot(
        x="system",
        y=[
            "density_accuracy",
            "density_macro_f1"
        ],
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)

    ax.set_title(
        "Traffic Density Classification"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()

    fig.savefig(
        SAVE_DIR / "density_comparison.png",
        dpi=150
    )

    plt.close(fig)

    # Generate an offline demonstration using saved detections.
    print("\nCreating traffic demonstration")

    sequence = test_manifest["sequence"].iloc[0]

    demo_frames = (
        test_manifest[
            test_manifest["sequence"] == sequence
        ]
        .sort_values("frame")
        .head(60)
    )

    report_lookup = reports.set_index(
        "image"
    )["final_report"].to_dict()

    animation = []

    for row in demo_frames.itertuples():

        record = lookup[row.image]

        with Image.open(
            image_paths[row.image]
        ) as original:

            image = original.convert("RGB")

        image = draw_boxes(
            image,
            selected_groups.get(
                row.image,
                chosen.iloc[:0]
            ),
            "lime"
        )

        canvas = Image.new(
            "RGB",
            (
                image.width,
                image.height + 120
            ),
            "black"
        )

        canvas.paste(image, (0, 0))

        draw = ImageDraw.Draw(canvas)

        draw.text(
            (10, image.height + 8),
            f"Vehicles: {record['vehicle_count']} | "
            f"Density: {record['density']} | "
            f"Condition: {record['condition']}",
            fill="white"
        )

        draw.text(
            (10, image.height + 28),
            f"Confidence: {record['confidence_tier']} | "
            f"Alert: {record['alert']}",
            fill="white"
        )

        report_text = report_lookup.get(
            row.image,
            template_report(record)
        )

        report_source = (
            "Verified LLM / fallback report"
            if row.image in report_lookup
            else "Template report"
        )

        draw.text(
            (10, image.height + 48),
            report_source,
            fill="white"
        )

        # Wrap long reports to fit within the image.
        words = report_text.split()

        lines = []
        current_line = ""

        for word in words:

            candidate = (
                current_line + " " + word
            ).strip()

            if draw.textlength(candidate) <= image.width - 20:

                current_line = candidate

            else:

                lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        for index, line in enumerate(lines):

            draw.text(
                (
                    10,
                    image.height + 68 + index * 16
                ),
                line,
                fill="white"
            )

        animation.append(canvas)

    if animation:

        animation[0].save(
            SAVE_DIR / "traffic_demo.gif",
            save_all=True,
            append_images=animation[1:],
            duration=200,
            loop=0
        )

    print("\nTraffic reporting complete.")
    print("Results saved to:", SAVE_DIR)


if __name__ == "__main__":
    main()
