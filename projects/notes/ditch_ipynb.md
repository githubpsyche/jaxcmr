gah. ive got to ditch jupyter notebooks iin my workflow but i still don't know if i have the heart


I hear you—once the JSON gets tweaked it’s easy to trip over notebook plumbing. If you want a middle ground, you can keep the code in plain .py or .md files and run parameterized fits via a small driver script (e.g., using YAML/JSON configs) instead of editing notebooks. Let me know if you’d like me to sketch that workflow or extract the key cells into a Python runner so you can avoid notebook fragility.

i like papermill as a way of generating scripts that corresponding with specific parametrizations of a template script. wouldn't be hard to custom-make something compatible with percent:py organized scripts, but i wonder if there are any pre-existing libraries?


A couple of off‑the‑shelf options fit the “parameterized script, not notebook JSON” niche:

Jupytext + Papermill: keep the notebook paired to a percent/light .py file. Edit/run the .py; Jupytext handles the round‑trip to .ipynb, and Papermill can still parameterize/run the notebook (use a pre‑convert step or the --jupytext plugin). This is the closest drop‑in to your current flow.
ploomber‑engine: can execute notebooks and .py files written with # %%/percent format as if they were notebooks. It also ships an engine so Papermill can run notebook‑formatted .py scripts without conversion.
Quarto/RMarkdown‑style params: Quarto supports params: for .qmd (and .py chunks), but that’s a bigger shift.
If you want to stick with Papermill, Jupytext pairing is the least disruptive: add a Jupytext header or .jupytext config, edit in the .py, let Papermill run the paired notebook. If you don’t want pairing files at all, ploomber‑engine lets you run the percent script directly.

i dont want to create or maintain ipynb files in a scenario where i ditch them. this made me want to reject the jupytext + papermill idea by default. but maybe there's a way to link the two without actually instantiating notebook files? for example, i suspect that there's a way for jupytext to store in a variable the output of converting a py file into ipynb


You don’t have to keep .ipynb inputs around. Two workable patterns:

In‑memory Jupytext → Papermill

Keep the notebook source as a py:percent file.
Read and convert in memory, then execute:
import jupytext, papermill as pm, tempfile

text = Path("template.py").read_text()
nb = jupytext.reads(text, fmt="py:percent")  # no .ipynb on disk

with tempfile.NamedTemporaryFile(suffix=".ipynb") as tmp:
    pm.execute_notebook(nb, tmp.name, parameters={"foo": 1})
    # tmp file is discarded
papermill.execute_notebook accepts a NotebookNode, so you never persist the input notebook; you can also drop the output to a temp path if you don’t want to keep it.

Use a Papermill engine that runs py:percent directly (no conversion step).

ploomber-engine provides this: pip install ploomber-engine
Then papermill template.py out.ipynb --engine=ploomber (or point the output to a temp file if you don’t need it).
Both approaches avoid maintaining .ipynb inputs; you can still parameterize runs via Papermill. The only artifact Papermill insists on is an output path—send it to a temp file if you don’t want to keep it.

ok reasonably clear that i can build a whole pipeline in py:percent format. does py:percent format have an established procedure for marking cells as parameters like ipynb does?


In py:percent, you add a parameters tag to the cell header, same as in a notebook. Example:

# %% [markdown]
# # Title

# %% tags=["parameters"]
my_param = 1
other = "default"

# %% code cell
result = my_param + 1
print(result)
Papermill (and any tool honoring nbformat tags) will treat that cell as the parameter cell after Jupytext conversion. So the established way is still tags=["parameters"] on the cell fence.

does jupytext preserve tags? like if i called it right now to convert my notebook w a param tag into py:perecent would the tag be preserved


Yes—Jupytext round‑trips cell metadata, including tags. If you convert your current notebook to py:percent the tags=["parameters"] will be preserved in the fenced cell header. Converting back to .ipynb restores the tags as well.

ok remaining thing missing from notebooks is the output log plus their arrangement within the notebook


Two pieces a .py:percent file won’t capture by itself:

Cell outputs/logs: Papermill only writes outputs into the output notebook. With percent scripts, you can still have Papermill write an output .ipynb (even to a temp file) to archive outputs; or redirect stdout/stderr to a log file when you run the script (e.g., papermill template.py /tmp/run.ipynb --engine=ploomber > run.log 2>&1). If you don’t want the .ipynb, you can discard it and keep just the log.

Cell ordering/sectioning: The # %% fences already define cell order; Markdown cells persist in the percent format as # %% [markdown] blocks. If you need a more “notebook-like” rendered layout, you can view the percent file in an editor that understands code cells, or render to HTML (via nbconvert) from the temporary .ipynb Papermill produces.

So you can keep code in percent format, tag the parameter cell, and still run with Papermill/engine—saving an output notebook (or HTML) when you need logs/outputs, or dumping stdout/stderr to a separate log otherwise.

hmm this feels quite elaborate. let's keep pushing. i could define a helper function that wraps around the papermill call i already use in my render notebooks with the key difference that the wrapper function takes py:percent files as their notebook argument instead of an ipynb. This way, I can ditch needing notebooks as the source file for my template notebooks, cool. Then I can decide whether I want to generate and execute an ipynb from that wrapper, another py:percent file, or both (paired jupytext-like). render notebooks themselves could easily be py:percent files. jupytext vscode extensions allow seeing notebook ui for percent:py files. so main reason to keep notebooks is to preserve outputs within the notebook after applicable cells when the notebook is closed. which itself is only relevant when outputs can't be easily re-generated when i re-open the notebook. notebooks like projects/TalmiEEG/notebooks/fitting_TalmiEEG_EEGMainEffects_50_set_likelihood_fixed_term_best_of_3.ipynb already preserve most outputs and avoid re-computations when outputs are preserved unless directly parametrized to re-run operations. the only thing i miss when i re-run is the progress log that later reminds me how long the fitting process took for each subject and overall. so maybe i just need to figure out how to store that log in a text file or something instead of rendering it in the cell


You can get what you want with a thin wrapper and external logging:

Wrap Papermill to accept a py:percent template and optionally emit an .ipynb and/or .py:

import tempfile, jupytext, papermill as pm
from pathlib import Path

def run_percent(template_py: Path, parameters: dict, save_ipynb: Path | None = None, log_path: Path | None = None):
    text = template_py.read_text()
    nb = jupytext.reads(text, fmt="py:percent")

    out_nb = save_ipynb or Path(tempfile.mkstemp(suffix=".ipynb")[1])
    pm.execute_notebook(
        nb, out_nb,
        parameters=parameters,
        log_output=True,
        stdout_file=str(log_path) if log_path else None,
        stderr_file=str(log_path) if log_path else None,
    )
    if not save_ipynb:
        out_nb.unlink(missing_ok=True)
This keeps your source as py:percent, preserves the parameter cell tag, and you can choose to keep or discard the output notebook.

For the progress log, use Papermill’s stdout_file/stderr_file (or subprocess redirection) to capture the cell output stream to a .log alongside the run. That way you don’t rely on cell outputs for timing info.

If you need a notebook view, Jupytext/VScode can render the py:percent file directly; only keep the output .ipynb when you specifically want inline outputs preserved.

This lets you drop .ipynb as your templates, keep param cells in percent scripts, and still archive execution logs and (optionally) output notebooks.

ive never used stdout_file arg before. how does it play with case where notebook cell includes something like plt.show()


stdout_file/stderr_file in papermill only capture the text streams sent to stdout/stderr while the notebook executes. They won’t capture display outputs like figures.

print, logging to stdout → goes into the log file when stdout_file is set.
plt.show() (or any display hook output) → remains a rich output in the output notebook; nothing is written to stdout_file.
Exceptions → still appear in the notebook output; if they produce a traceback on stderr, stderr_file will get that text.
So you can use stdout_file to keep the progress/timing text, and still rely on the output notebook (or Jupytext view) for images/plots produced by plt.show().

Help me decide whether I want to go through with this. What are the pros of ditching ipynb


Pros of ditching .ipynb as your primary source:

Version control sanity: Clean diffs/merges. No execution counts, metadata noise, or hidden binary blobs; easier to review and bisect.
Less fragility: No JSON corruption issues like the token errors you’ve hit. Plain text .py is harder to break and easier to repair.
Editor flexibility: Treat them like normal code; use your usual tooling (lint/format, grep/rg, code review) without notebook-specific quirks.
Parameterization without GUI: With Jupytext/py:percent + Papermill (or a wrapper), you can still parameterize runs, but your “notebook” is just code.
Reproducible logging: You can redirect stdout/stderr to log files; don’t rely on inline cell outputs for timing/progress.
Batch/CI friendly: Straightforward to run in headless environments; no need to pre‑bake .ipynb inputs.
Cons/what you lose:

Inline rich outputs aren’t preserved in the source file; you need an output artifact (notebook/HTML) if you want a saved view of figures.
Less immediate GUI interactivity unless your editor renders percent scripts as notebooks.
Slightly more scaffolding (e.g., a small runner script) to emulate the “Run All” experience with parameters.
Given your pain points (invalid notebook JSON, desire to keep percent format), moving templates to py:percent and only generating .ipynb outputs when needed gives you most of the notebook conveniences with far fewer pitfalls.