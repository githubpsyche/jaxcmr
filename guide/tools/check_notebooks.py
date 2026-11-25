"""Run bounded notebook copies with nbmake in the current project environment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "guide"
# Test cases and workload profiles, not notebook installation requirements.
CASES = [
    {'path': 'templates/fitting.ipynb', 'profile': 'fitting'},
    {'path': 'templates/simulation.ipynb', 'profile': 'simulation'},
    {'path': 'templates/parameter_shifting.ipynb', 'profile': 'shifting'},
    {'path': 'templates/model_comparison.ipynb', 'profile': 'comparison'},
    {'path': 'templates/spc.ipynb', 'profile': 'analysis'},
    {'path': 'templates/pnr.ipynb', 'profile': 'analysis'},
    {'path': 'templates/termination_probability.ipynb', 'profile': 'analysis'},
    {'path': 'templates/crp.ipynb', 'profile': 'analysis'},
    {'path': 'templates/lagrank.ipynb', 'profile': 'analysis'},
    {'path': 'templates/distcrp.ipynb', 'profile': 'analysis'},
    {'path': 'templates/distrank.ipynb', 'profile': 'analysis'},
    {'path': 'templates/cat_recall_rate.ipynb', 'profile': 'analysis'},
    {'path': 'templates/cat_spc.ipynb', 'profile': 'analysis'},
    {'path': 'templates/catcrp.ipynb', 'profile': 'analysis'},
    {'path': 'guide/examples/first-simulation.ipynb', 'profile': 'first'},
    {'path': 'guide/tutorials/data-preparation.ipynb', 'profile': 'preparation'},
    {'path': 'guide/examples/war-photo-data-preparation.ipynb', 'profile': 'spreadsheet'},
]
WINDOWS_CASES = {"first-simulation", "simulation", "spc", "war-photo-data-preparation"}


def overrides(profile, project):
    common = f"project_root = {str(project)!r}\noutput_dir = 'results'\n"
    settings = {
        "analysis": "max_subjects = 4\ndata_paths = data_paths[:2]\ndata_labels = data_labels[:2]\ntrial_queries = trial_queries[:2]\nlabels = labels[:2] if labels is not None else None\n",
        "simulation": "max_subjects = 2\nexperiment_count = 1\nsimulation_dir = 'results/responses'\n",
        "fitting": "fit_action = 'estimate'\nmax_subjects = 2\nmodel_configs = {'CMR': model_configs['CMR']}\nfit_hyperparams.update(num_steps=2, pop_size=2, best_of=1)\n",
        "shifting": "max_subjects = 2\nexperiment_count = 1\nparameter_sweeps = {'CMR_encoding': dict(parameter_sweeps['CMR_encoding'], values=[0.2, 0.8])}\n",
        "comparison": "", "first": "", "preparation": "",
        "spreadsheet": "source_dir = 'synthetic-inputs'\n",
    }[profile]
    if profile in {"simulation", "shifting", "comparison"}:
        settings += "comparison_analysis_configs = [c for c in comparison_analysis_configs if c['target'] == 'jaxcmr.analyses.spc.plot_spc']\nassert len(comparison_analysis_configs) == 1\n"
    return common + settings


def prepared_case(raw, item, project):
    import nbformat

    notebook = nbformat.reads(raw.decode(), as_version=4)
    candidates = []
    for c in notebook.cells:
        if c.cell_type != "code":
            continue
        tags = c.metadata.get("tags", [])
        if set(tags) & {"skip-execution", "raises-exception"} or c.metadata.get("nbmake"):
            raise ValueError(f"Required execution cannot be skipped or overridden: {item['path']}")
        c.outputs = []; c.execution_count = None
        c.metadata.pop("execution", None)
        if "parameters" in tags or "project_root =" in c.source:
            candidates.append(c)
    if item["profile"] != "first":
        if len(candidates) != 1:
            raise ValueError(f"Expected one parameter cell: {item['path']}")
        candidates[0].metadata["nbmake"] = {"post_cell_execute": [overrides(item["profile"], project)]}
    notebook.metadata["execution"] = {"timeout": 600, "allow_errors": False}
    notebook.cells.insert(0, nbformat.v4.new_code_cell("import _reader_assertions as _checks\n_checks.start()", id="reader-start"))
    argument = "recalls" if item["profile"] == "first" else ""
    notebook.cells.append(nbformat.v4.new_code_cell(f"_checks.finish({argument})", id="reader-output-check"))
    return notebook


def clean_env():
    env = dict(os.environ)
    for key in ["PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "JUPYTER_PATH", "PYTEST_ADDOPTS"]:
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               MPLBACKEND="module://matplotlib_inline.backend_inline", JAX_PLATFORMS="cpu",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def junit_passed(path, expected):
    if not path.exists():
        return False
    tests = ET.parse(path).findall(".//testcase")
    return len(tests) == expected and all(not any(c.find(tag) is not None for tag in ("failure", "error", "skipped")) for c in tests)


def completed_notebook(notebook):
    return all(c.execution_count is not None for c in notebook.cells
               if c.cell_type == "code" and c.source.strip())


def execute(items, work, jobs=2):
    import nbformat

    started = time.perf_counter()
    work.mkdir(parents=True, exist_ok=True)
    # Use this interpreter explicitly, independent of any user-installed kernels.
    kernel = work / "jupyter/kernels/jaxcmr-reader"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "JAXCMR notebook checks", "language": "python",
    }), encoding="utf-8")
    env = clean_env()
    env["JUPYTER_PATH"] = str(work / "jupyter")
    projects = []
    from spreadsheet_fixture import create
    from reader_assertions import environment
    for item in items:
        project = work / Path(item["path"]).stem
        project.mkdir()
        (project / "case.json").write_text(json.dumps(item), encoding="utf-8")
        shutil.copy2(GUIDE / "tools/reader_assertions.py", project / "_reader_assertions.py")
        raw = (ROOT / item["path"]).read_bytes()
        nbformat.write(prepared_case(raw, item, project), project / "executed.ipynb")
        if item["profile"] == "spreadsheet":
            create(project / "synthetic-inputs")
        projects.append((item, project))
    report = {"environment": environment(), "jobs": jobs,
              "preparation_seconds": round(time.perf_counter() - started, 3)}
    config = work / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    command = [sys.executable, "-m", "pytest", "-p", "nbmake.pytest_plugin", "-p", "xdist.plugin",
               "--nbmake", "--overwrite", "--nbmake-kernel=jaxcmr-reader", "--nbmake-timeout=600",
               "-n", str(jobs), "--rootdir", str(work), "-c", str(config),
               "--junitxml", str(work / "junit.xml"), "--durations=20", "-q",
               *[str(p / "executed.ipynb") for _, p in projects]]
    running = time.perf_counter()
    result = subprocess.run(command, cwd=work, env=env)
    report["execution_seconds"] = round(time.perf_counter() - running, 3)
    passed = result.returncode == 0 and junit_passed(work / "junit.xml", len(projects))
    report["cases"] = []
    for item, project in projects:
        path = project / "result.json"
        metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "failed"}
        executed = nbformat.read(project / "executed.ipynb", as_version=4)
        if not completed_notebook(executed):
            metrics["status"] = "failed"
        passed &= metrics["status"] == "passed"
        report["cases"].append({**item, **metrics})
    report["wall_seconds"] = round(time.perf_counter() - started, 3)
    report["passed"] = passed
    (work / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Notebook suite: {report['wall_seconds']} seconds; report: {work / 'report.json'}", flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--select", nargs="+")
    parser.add_argument("--platform", choices=["all", "windows"], default="all")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    items = CASES
    if args.select:
        items = [i for i in items if i["path"] in args.select or Path(i["path"]).stem in args.select]
        if len(items) != len(set(args.select)):
            parser.error("Unknown or duplicated notebook selection")
    if args.platform == "windows":
        items = [i for i in items if Path(i["path"]).stem in WINDOWS_CASES]
    if not items:
        parser.error("No notebook cases selected")
    work = args.work.resolve() if args.work else Path(tempfile.mkdtemp(prefix="jaxcmr-notebooks-"))
    if work.is_relative_to(ROOT):
        parser.error("Use a temporary folder outside the checkout for notebook execution")
    raise SystemExit(0 if execute(items, work, jobs=args.jobs) else 1)


if __name__ == "__main__":
    main()
