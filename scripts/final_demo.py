
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from detector_fusion import load_manifest


ROOT = Path(__file__).resolve().parents[1]

FUSION = ROOT / "outputs" / "fusion"
REPORTING = FUSION / "reporting"
COMPARISON = FUSION / "comparison"
SAVE_DIR = FUSION / "final_demo"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = [
    "baseline",
    "p2",
    "frcnn",
    "fusion_static",
    "fusion_condition",
    "fusion_fast"
]

NAMES = {
    "baseline": "Baseline YOLO",
    "p2": "YOLO + P2",
    "frcnn": "Faster R-CNN",
    "fusion_static": "Static fusion",
    "fusion_condition": "Condition-aware fusion",
    "fusion_fast": "Fast fusion"
}

COLORS = {
    "baseline": "#808080",
    "p2": "#2563eb",
    "frcnn": "#ea580c",
    "fusion_static": "#9333ea",
    "fusion_condition": "#16a34a",
    "fusion_fast": "#0891b2"
}

UNCERTAINTY = {
    "high": "Detector agreement is relatively strong.",
    "medium": "The vehicle count is an approximate estimate.",
    "low": "The vehicle count is uncertain."
}

FACTUAL_FIELDS = [
    "vehicle_count",
    "density",
    "confidence_tier",
    "alert"
]


def load_records():

    records = json.loads(
        (REPORTING / "traffic_records.json").read_text()
    )

    return {
        row["image"]: row
        for row in records
    }


# Evaluate core factual consistency separately
# from strict output verification.
def evaluate_report_claims(records):

    experiments = {
        "original_json": (
            REPORTING / "report_evaluation_original.csv"
        ),
        "improved_prompt_schema": (
            REPORTING / "report_evaluation.csv"
        )
    }

    summary = []
    details = []

    for method, path in experiments.items():

        reports = pd.read_csv(path)

        completed = 0
        parsed_reports = 0
        factual_errors = 0
        factual_fields = 0
        missing_fields = 0
        extra_fields = 0
        reports_with_errors = 0
        uncertainty_errors = 0
        strict_passes = 0

        for row in reports.itertuples(index=False):

            if row.status != "completed":
                continue

            completed += 1

            if str(row.llm_supported).lower() == "true":
                strict_passes += 1

            record = records[row.image]

            try:

                parsed = json.loads(row.llm_raw)

                if not isinstance(parsed, dict):
                    raise ValueError("Expected JSON object")

            except (ValueError, TypeError):

                details.append({
                    "method": method,
                    "image": row.image,
                    "valid_json": False
                })

                continue

            parsed_reports += 1

            incorrect = []
            missing = []

            for field in FACTUAL_FIELDS:

                if field not in parsed:

                    missing.append(field)

                    continue

                factual_fields += 1

                correct = (
                    parsed[field] == record[field]
                )

                if field == "vehicle_count":

                    correct = (
                        correct
                        and type(parsed[field]) is int
                    )

                if not correct:

                    incorrect.append(field)

            unexpected = set(parsed) - set(
                FACTUAL_FIELDS + ["uncertainty"]
            )

            factual_errors += len(incorrect)
            missing_fields += len(missing)
            extra_fields += len(unexpected)

            if incorrect:
                reports_with_errors += 1

            uncertainty_correct = (
                parsed.get("uncertainty")
                == UNCERTAINTY[
                    record["confidence_tier"]
                ]
            )

            if not uncertainty_correct:
                uncertainty_errors += 1

            details.append({
                "method": method,
                "image": row.image,
                "valid_json": True,
                "incorrect_core_fields": len(incorrect),
                "missing_core_fields": len(missing),
                "unexpected_fields": len(unexpected),
                "incorrect_field_names": ",".join(incorrect),
                "missing_field_names": ",".join(missing),
                "uncertainty_correct": uncertainty_correct,
                "strict_pass": (
                    str(row.llm_supported).lower() == "true"
                )
            })

        summary.append({
            "method": method,
            "attempted_reports": len(reports),
            "completed_reports": completed,
            "valid_json_reports": parsed_reports,
            "evaluated_core_fields": factual_fields,
            "incorrect_core_fields": factual_errors,
            "missing_core_fields": missing_fields,
            "unexpected_fields": extra_fields,
            "core_field_error_rate": (
                factual_errors / factual_fields
                if factual_fields else np.nan
            ),
            "reports_with_core_errors": reports_with_errors,
            "uncertainty_errors": uncertainty_errors,
            "strict_verified_reports": strict_passes,
            "strict_verification_rate": (
                strict_passes / completed
                if completed else np.nan
            )
        })

    table = pd.DataFrame(summary)

    table.to_csv(
        SAVE_DIR / "claim_evaluation.csv",
        index=False
    )

    pd.DataFrame(details).to_csv(
        SAVE_DIR / "claim_evaluation_details.csv",
        index=False
    )

    print("\nLLM factual-field evaluation:")

    print(
        table.to_string(index=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    values = table[
        "strict_verification_rate"
    ] * 100

    ax.bar(
        [
            "Original JSON prompt",
            "Improved prompt + JSON schema"
        ],
        values,
        color=["#808080", "#2563eb"]
    )

    for index, value in enumerate(values):

        ax.text(
            index,
            value + 2,
            f"{value:.1f}%",
            ha="center",
            fontsize=12
        )

    ax.set_ylim(0, 110)

    ax.set_ylabel(
        "Strict verification rate (%)"
    )

    ax.set_title(
        "LLM Report Verification"
    )

    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    fig.savefig(
        SAVE_DIR / "llm_verification.png",
        dpi=180
    )

    plt.close(fig)


# Evaluate the distribution of density categories.
def evaluate_density_distribution():

    records = pd.read_csv(
        REPORTING / "traffic_records.csv"
    )

    thresholds = json.loads(
        (REPORTING / "density_thresholds.json").read_text()
    )

    labels = [
        "low",
        "medium",
        "high"
    ]

    actual = pd.cut(
        records["ground_truth_count"],
        bins=[
            -1,
            thresholds["low_max"],
            thresholds["medium_max"],
            np.inf
        ],
        labels=labels
    )

    predicted = records["density"]

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=labels
    )

    table = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    table.index.name = "actual"

    table.to_csv(
        SAVE_DIR / "density_confusion_matrix.csv"
    )

    report = classification_report(
        actual,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).T.to_csv(
        SAVE_DIR / "density_class_report.csv"
    )

    print("\nDensity confusion matrix:")

    print(table)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(
        matrix,
        cmap="Blues"
    )

    ax.set_xticks(
        range(3),
        labels
    )

    ax.set_yticks(
        range(3),
        labels
    )

    for i in range(3):

        for j in range(3):

            value = matrix[i, j]

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > matrix.max() / 2
                    else "black"
                ),
                fontsize=14
            )

    ax.set_xlabel("Predicted density")
    ax.set_ylabel("Actual density")

    ax.set_title(
        "Condition-Aware Fusion: Density Classification"
    )

    plt.tight_layout()

    fig.savefig(
        SAVE_DIR / "density_confusion_matrix.png",
        dpi=180
    )

    plt.close(fig)


# Create a new recall plot using six distinct colors.
def plot_recall():

    results = pd.read_csv(
        COMPARISON / "error_bars_by_video.csv"
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, 5),
        sharey=True
    )

    panels = [
        (
            "Recall by vehicle size",
            ["small", "medium"]
        ),
        (
            "Recall by occlusion",
            [
                "partial_occlusion",
                "heavy_occlusion"
            ]
        )
    ]

    for ax, (title, subsets) in zip(
        axes,
        panels
    ):

        groups = [
            (subset, condition)
            for subset in subsets
            for condition in ["day", "low_light"]
        ]

        width = 0.8 / len(SYSTEMS)

        for index, model in enumerate(SYSTEMS):

            values = []
            lower = []
            upper = []

            for subset, condition in groups:

                row = results[
                    (results["metric"] == "recall")
                    & (results["subset"] == subset)
                    & (results["condition"] == condition)
                    & (results["system"] == model)
                ].iloc[0]

                values.append(row["value"])

                lower.append(
                    max(
                        0,
                        row["value"] - row["ci_low"]
                    )
                )

                upper.append(
                    max(
                        0,
                        row["ci_high"] - row["value"]
                    )
                )

            x = (
                np.arange(len(groups))
                + (
                    index
                    - (len(SYSTEMS) - 1) / 2
                ) * width
            )

            ax.bar(
                x,
                values,
                width,
                yerr=[lower, upper],
                capsize=2,
                color=COLORS[model],
                label=NAMES[model]
            )

        ax.set_xticks(
            np.arange(len(groups))
        )

        ax.set_xticklabels([
            (
                subset.replace("_occlusion", "")
                + "\n"
                + condition.replace("_", " ")
            )
            for subset, condition in groups
        ])

        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel(
        "Vehicle recall (confidence 0.25, IoU 0.5)"
    )

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.87]
    )

    fig.savefig(
        SAVE_DIR / "recall_comparison.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


# Compare fixed and validation-tuned counting thresholds.
def plot_counting():

    table = pd.read_csv(
        COMPARISON / "count_tuned_threshold.csv"
    )

    overall = (
        table[table["condition"] == "all"]
        .set_index("system")
        .loc[SYSTEMS]
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    x = np.arange(len(SYSTEMS))

    width = 0.36

    ax.bar(
        x - width / 2,
        overall["count_MAE @0.25"],
        width,
        label="Common threshold: 0.25",
        color="#2563eb"
    )

    ax.bar(
        x + width / 2,
        overall["count_MAE tuned"],
        width,
        label="Validation-tuned thresholds",
        color="#ea580c"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [NAMES[m] for m in SYSTEMS],
        rotation=20,
        ha="right"
    )

    ax.set_ylabel(
        "Vehicle counting MAE"
    )

    ax.set_title(
        "Vehicle Counting Performance"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    fig.savefig(
        SAVE_DIR / "counting_comparison.png",
        dpi=180
    )

    plt.close(fig)


def get_font(size):

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "DejaVuSans.ttf"
    ]

    for path in candidates:

        try:

            return ImageFont.truetype(
                path,
                size
            )

        except OSError:

            continue

    return ImageFont.load_default()


def wrap_text(text, draw, font, max_width):

    lines = []
    line = ""

    for word in text.split():

        candidate = (
            line + " " + word
        ).strip()

        if draw.textlength(
            candidate,
            font=font
        ) <= max_width:

            line = candidate

        else:

            if line:
                lines.append(line)

            line = word

    if line:
        lines.append(line)

    return lines


# Create an offline demonstration using saved
# detector outputs and actual verified LLM responses.
def build_demo(records):

    settings = json.loads(
        (FUSION / "fusion_settings.json").read_text()
    )

    reporting_settings = json.loads(
        (REPORTING / "reporting_settings.json").read_text()
    )

    selected = reporting_settings[
        "selected_system"
    ]

    threshold = reporting_settings[
        "thresholds"
    ][selected]

    detections = pd.read_csv(
        FUSION / f"{selected}_test.csv"
    )

    detections = detections[
        detections["confidence"] >= threshold
    ].copy()

    groups = dict(
        tuple(detections.groupby("image"))
    )

    manifest = load_manifest("test")

    reports = pd.read_csv(
        REPORTING / "report_evaluation.csv"
    )

    verified = reports[
        reports["llm_supported"]
        .astype(str)
        .str.lower()
        .eq("true")
    ].copy()

    font = get_font(20)
    heading = get_font(27)
    small_font = get_font(17)

    animation = []
    examples = []

    for condition in ["day", "night"]:

        candidates = verified[
            verified["condition"] == condition
        ]

        if candidates.empty:

            print(
                f"No verified LLM example for {condition}"
            )

            continue

        # Use the first verified report for each condition.
        # This is an illustrative example, not an evaluation subset.
        report_row = candidates.iloc[0]

        image_id = report_row["image"]

        anchor = manifest[
            manifest["image"] == image_id
        ].iloc[0]

        sequence = anchor["sequence"]

        frames = (
            manifest[
                manifest["sequence"] == sequence
            ]
            .sort_values("frame")
            .reset_index(drop=True)
        )

        positions = np.flatnonzero(
            frames["image"].eq(image_id).to_numpy()
        )

        position = int(positions[0])

        start = max(0, position - 15)

        frames = frames.iloc[
            start:start + 35
        ]

        frame_number = 0

        for row in frames.itertuples(index=False):

            record = records[row.image]

            image_path = (
                ROOT / "data" / "processed"
                / "images" / "test"
                / Path(row.image_path).name
            )

            with Image.open(image_path) as original:

                image = original.convert("RGB")

            draw = ImageDraw.Draw(image)

            boxes = groups.get(
                row.image,
                detections.iloc[:0]
            )

            for box in boxes.itertuples(index=False):

                draw.rectangle(
                    [
                        box.x1,
                        box.y1,
                        box.x2,
                        box.y2
                    ],
                    outline="#35ef70",
                    width=3
                )

            canvas = Image.new(
                "RGB",
                (
                    image.width,
                    image.height + 220
                ),
                "#111827"
            )

            canvas.paste(
                image,
                (0, 0)
            )

            draw = ImageDraw.Draw(canvas)

            lighting = record["condition"]

            weights = settings[
                "night_weights"
                if lighting == "night"
                else "day_weights"
            ]

            weights_text = " : ".join(
                str(int(w))
                for w in weights
            )

            y = image.height + 10

            draw.text(
                (15, y),
                f"{lighting.upper()} | Weights: {weights_text}",
                font=heading,
                fill="#ffffff"
            )

            draw.text(
                (15, y + 40),
                (
                    f"Vehicles: {record['vehicle_count']}   "
                    f"Density: {record['density']}   "
                    f"Confidence: {record['confidence_tier']}"
                ),
                font=font,
                fill="#ffffff"
            )

            agreement = record[
                "detector_agreement"
            ]

            agreement_text = (
                f"{agreement:.1%}"
                if agreement is not None
                else "N/A"
            )

            draw.text(
                (15, y + 70),
                (
                    f"Detector agreement: {agreement_text}   "
                    f"Alert: {record['alert']}"
                ),
                font=small_font,
                fill="#d1d5db"
            )

            if row.image == image_id:

                text = report_row["final_report"]

                source = "Verified LLM report"

            else:

                text = (
                    f"Estimated vehicles: "
                    f"{record['vehicle_count']}. "
                    f"Scene density: "
                    f"{record['density']}. "
                    f"Confidence tier: "
                    f"{record['confidence_tier']}."
                )

                source = "Deterministic template report"

            draw.text(
                (15, y + 100),
                source,
                font=font,
                fill="#7dd3fc"
            )

            lines = wrap_text(
                text,
                draw,
                small_font,
                image.width - 30
            )

            for index, line in enumerate(lines):

                draw.text(
                    (
                        15,
                        y + 130 + index * 22
                    ),
                    line,
                    font=small_font,
                    fill="#ffffff"
                )

            animation.append(canvas)

            if row.image == image_id:

                # Hold verified examples long enough to read.
                for _ in range(14):

                    animation.append(
                        canvas.copy()
                    )

                canvas.save(
                    SAVE_DIR
                    / f"verified_{condition}_example.png"
                )

                examples.append({
                    "image": image_id,
                    "condition": lighting,
                    "sequence": sequence,
                    "weights": weights,
                    "report": text,
                    "demo_type": "precomputed detections"
                })

            frame_number += 1

        print(
            f"{condition} demo: {frame_number} frames"
        )

    if not animation:

        raise RuntimeError(
            "No demonstration frames generated"
        )

    animation[0].save(
        SAVE_DIR / "traffic_demo_day_night.gif",
        save_all=True,
        append_images=animation[1:],
        duration=200,
        loop=0
    )

    (SAVE_DIR / "demo_examples.json").write_text(
        json.dumps(
            examples,
            indent=2
        )
    )

    print("\nDay/night demonstration generated.")


def main():

    records = load_records()

    evaluate_report_claims(records)

    evaluate_density_distribution()

    plot_recall()

    plot_counting()

    build_demo(records)

    print("\nFinal review complete.")

    print(
        "Outputs saved to:",
        SAVE_DIR
    )


if __name__ == "__main__":

    main()
