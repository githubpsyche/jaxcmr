"""Checks copied into temporary reader projects; never imported by published notebooks."""
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import time


def environment():
    return {"python": sys.version, "executable": sys.executable,
            "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())}


def start():
    global _started
    _started = time.perf_counter()


def verify_outputs(project, profile):
    import numpy as np
    import pandas as pd
    from PIL import Image
    from jaxcmr.helpers import load_data
    root = Path(project) / "results"
    files = [p for p in root.rglob("*") if p.is_file()]
    assert files, "Notebook did not save any results"
    for path in files:
        if path.suffix == ".png":
            with Image.open(path) as picture:
                picture.verify()
            with Image.open(path) as picture:
                picture.load()
                assert min(picture.size) > 0
        elif path.suffix == ".csv":
            assert len(pd.read_csv(path)) > 0, path
        elif path.suffix == ".json":
            assert isinstance(json.loads(path.read_text(encoding="utf-8")), (dict, list)), path
        elif path.suffix == ".h5":
            data = load_data(str(path))
            assert len(data["recalls"]) > 0 and np.isfinite(data["recalls"]).all(), path
        elif path.suffix == ".npy":
            assert np.isfinite(np.load(path)).all(), path
    if profile == "preparation":
        assert len(load_data(str(root / "teaching_recall.h5"))["subject"]) == 3
    elif profile == "spreadsheet":
        data = load_data(str(root / "Dupertuys2026.h5"))
        assert len(data["subject"]) == 492
        np.testing.assert_array_equal(data["recalls"][0, :4], [1, 2, 3, 4])
    elif profile == "simulation":
        assert len(list((root / "responses").glob("*.h5"))) == 3
    elif profile == "fitting":
        assert any("fits" in json.loads(p.read_text(encoding="utf-8")) for p in (root / "fits").glob("*.json"))
    if profile in {"analysis", "simulation", "fitting", "shifting", "comparison"}:
        assert any(p.suffix == ".png" for p in files), "Missing figure"


def finish(recalls=None):
    case = json.loads(Path("case.json").read_text(encoding="utf-8"))
    elapsed = time.perf_counter() - _started
    started = time.perf_counter()
    if case["profile"] == "first":
        import numpy as np
        np.testing.assert_array_equal(np.asarray(recalls), [1, 5, 2, 4, 3])
    else:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--readback"], check=True, timeout=180)
    result = {"execution_seconds": round(elapsed, 3),
              "readback_seconds": round(time.perf_counter() - started, 3),
              "status": "passed"}
    Path("result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "--environment":
        Path(sys.argv[2]).write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    elif sys.argv[1] == "--readback":
        case = json.loads(Path("case.json").read_text(encoding="utf-8"))
        verify_outputs(Path.cwd(), case["profile"])
        print("Verified saved results in an independent Python process")
    else:
        raise SystemExit("Unknown reader assertion operation")
