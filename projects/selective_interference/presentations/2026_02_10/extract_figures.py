"""Extract PNG figures from executed Jupyter notebooks.

Reads notebook JSON, finds cells with image/png outputs, decodes
the base64 data, and saves as PNG files in the figures/ directory.
"""

import base64
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
CODE_DIR = SCRIPT_DIR.parent.parent / "code"
SIM_NOTEBOOK = CODE_DIR / "selective_interference_simulations.ipynb"


def load_notebook(path: Path) -> dict:
    """Load a notebook JSON file."""
    with open(path) as f:
        return json.load(f)


def find_cell_by_id(nb: dict, cell_id: str) -> dict | None:
    """Find a cell by its id field."""
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            return cell
    return None


def get_image_cells(nb: dict) -> list[dict]:
    """Return all cells that have at least one image/png output."""
    result = []
    for cell in nb["cells"]:
        for output in cell.get("outputs", []):
            if "image/png" in output.get("data", {}):
                result.append(cell)
                break
    return result


def extract_png(cell: dict, output_index: int = 0) -> bytes | None:
    """Extract PNG bytes from a cell's outputs."""
    img_count = 0
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        if "image/png" in data:
            if img_count == output_index:
                b64 = data["image/png"]
                if isinstance(b64, list):
                    b64 = "".join(b64)
                return base64.b64decode(b64)
            img_count += 1
    return None


def save_png(data: bytes, name: str) -> Path:
    """Save PNG bytes to the figures directory."""
    path = FIGURES_DIR / name
    path.write_bytes(data)
    return path


def extract_simulation_figures():
    """Extract key plots from the simulation notebook by cell ID."""
    nb = load_notebook(SIM_NOTEBOOK)

    targets = {
        "d7rzy585za9": "sim_baseline_spc.png",
        "lmaldg4r3u8": "sim1_mcf_sweep.png",
        "e82d4ebe": "sim1_drift_sweep.png",
        "6fdba71d": "sim1_count_sweep.png",
        "5166cd38": "sim1_summary.png",
        "tc59eeau29e": "sim3_tau_sweep.png",
        "4f3d0c9rlri": "sim3_interaction.png",
    }

    for cell_id, filename in targets.items():
        cell = find_cell_by_id(nb, cell_id)
        if cell is None:
            print(f"  SKIP {filename}: cell {cell_id} not found")
            continue
        data = extract_png(cell)
        if data is None:
            print(f"  SKIP {filename}: no image/png in cell {cell_id}")
            continue
        path = save_png(data, filename)
        print(f"  OK   {filename} ({len(data):,} bytes)")


def extract_analysis_figures():
    """Extract the first plot (Task main effect) from each analysis notebook."""
    notebooks = {
        "spc_report.ipynb": "data_spc_task.png",
        "lag_crp_report.ipynb": "data_lag_crp_task.png",
        "pnr_report.ipynb": "data_pnr_task.png",
        "cue_effectiveness_report.ipynb": "data_cue_eff_task.png",
    }

    # Also extract the second plot (Condition main effect) for cue effectiveness
    condition_notebooks = {
        "cue_effectiveness_report.ipynb": "data_cue_eff_condition.png",
        "spc_report.ipynb": "data_spc_condition.png",
    }

    for nb_name, filename in notebooks.items():
        nb_path = CODE_DIR / nb_name
        if not nb_path.exists():
            print(f"  SKIP {filename}: {nb_name} not found")
            continue
        nb = load_notebook(nb_path)
        image_cells = get_image_cells(nb)
        if not image_cells:
            print(f"  SKIP {filename}: no image cells in {nb_name}")
            continue
        data = extract_png(image_cells[0])  # first image = Task main effect
        if data is None:
            print(f"  SKIP {filename}: extraction failed")
            continue
        save_png(data, filename)
        print(f"  OK   {filename} ({len(data):,} bytes)")

    for nb_name, filename in condition_notebooks.items():
        nb_path = CODE_DIR / nb_name
        nb = load_notebook(nb_path)
        image_cells = get_image_cells(nb)
        if len(image_cells) < 2:
            print(f"  SKIP {filename}: fewer than 2 image cells")
            continue
        data = extract_png(image_cells[1])  # second image = Condition main effect
        if data is None:
            print(f"  SKIP {filename}: extraction failed")
            continue
        save_png(data, filename)
        print(f"  OK   {filename} ({len(data):,} bytes)")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Extracting simulation figures...")
    extract_simulation_figures()

    print("\nExtracting analysis figures...")
    extract_analysis_figures()

    print(f"\nDone. Figures in {FIGURES_DIR}")
