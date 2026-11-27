
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "notebooks" / "compare_detectors.ipynb"

SAVE_DIR = ROOT / "outputs" / "fusion"

DESTINATION = SAVE_DIR / "fusion_comparison.ipynb"

SYSTEM_NAMES = [
    "fusion_static",
    "fusion_condition",
    "fusion_fast"
]


def main():

    SAVE_DIR.mkdir(
        parents=True, exist_ok=True
    )

    # Confirm that the fusion experiments finished.
    for name in SYSTEM_NAMES:

        for split in ["val", "test"]:

            path = SAVE_DIR / f"{name}_{split}.csv"

            if not path.exists():
                raise FileNotFoundError(path)

    notebook = nbformat.read(
        SOURCE, as_version=4
    )

    changes = {
        "output": False,
        "test": False,
        "val": False
    }

    for cell in notebook.cells:

        if cell.cell_type != "code":
            continue

        source = cell.source

        # Redirect evaluation outputs to our own folder.
        old = "OUTPUT_DIR = OUT_ROOT / 'comparison'"

        if old in source:

            source = source.replace(
                old,
                "OUTPUT_DIR = OUT_ROOT / 'fusion' / 'comparison'"
            )

            changes["output"] = True

        # Include fused test predictions.
        if "ALL_SYSTEMS = {" in source:

            marker = "SYSTEMS = {k: p"

            if marker not in source:

                raise RuntimeError(
                    "Unexpected comparison notebook structure"
                )

            addition = """
FUSION_DIR = OUT_ROOT / 'fusion'

ALL_SYSTEMS.update({
    'fusion_static': FUSION_DIR / 'fusion_static_test.csv',
    'fusion_condition': FUSION_DIR / 'fusion_condition_test.csv',
    'fusion_fast': FUSION_DIR / 'fusion_fast_test.csv'
})

"""

            source = source.replace(
                marker,
                addition + marker,
                1
            )

            changes["test"] = True

        # Include fused validation predictions.
        if "VAL_FILES = {" in source:

            marker = "VAL_GT_FILE ="

            if marker not in source:

                raise RuntimeError(
                    "Validation configuration not found"
                )

            addition = """
VAL_FILES.update({
    'fusion_static': FUSION_DIR / 'fusion_static_val.csv',
    'fusion_condition': FUSION_DIR / 'fusion_condition_val.csv',
    'fusion_fast': FUSION_DIR / 'fusion_fast_val.csv'
})

"""

            source = source.replace(
                marker,
                addition + marker,
                1
            )

            changes["val"] = True

        # Skip the optional diagnostic if its CSV is absent.
        if (
            "import datetime" in source
            and "yolov10_metrics.csv" in source
        ):

            source = (
                "if (BASELINE_DIR / 'yolov10_metrics.csv').exists():\n"
                + "\n".join(
                    "    " + line
                    for line in source.splitlines()
                )
            )

        cell.source = source
        cell.outputs = []
        cell.execution_count = None

    if not all(changes.values()):

        raise RuntimeError(
            f"Notebook configuration changes incomplete: {changes}"
        )

    nbformat.write(
        notebook,
        DESTINATION
    )

    print("Running detector comparison notebook")
    print("Original notebook will remain unchanged.")

    client = NotebookClient(
        notebook,
        timeout=3600,
        kernel_name="traffic-fusion",
        resources={
            "metadata": {
                "path": str(ROOT)
            }
        }
    )

    try:

        client.execute()

    finally:

        nbformat.write(
            notebook,
            DESTINATION
        )

    print("\nEvaluation complete.")
    print("Notebook:", DESTINATION)
    print("Results:", SAVE_DIR / "comparison")


if __name__ == "__main__":
    main()
