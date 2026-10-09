"""Regenerate and execute the synthetic KESurv documentation notebook.

Run from the repository root after installing ``kesurv[docs,demo]``::

    python examples/build_synthetic_notebook.py
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "synthetic-demo.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-insecure-jupyter-writes", action="store_true",
        help="permit Jupyter connection-file writes when the OS denies permission changes",
    )
    args = parser.parse_args()
    if args.allow_insecure_jupyter_writes:
        os.environ["JUPYTER_ALLOW_INSECURE_WRITES"] = "1"

    import nbformat
    from nbclient import NotebookClient

    cells = [
        nbformat.v4.new_markdown_cell(
            "# Synthetic survival curves\n\n"
            "This executable notebook generates right-censored survival data, fits KESurv, "
            "and compares patient-specific survival curves with a population Kaplan–Meier "
            "curve. The random seed makes the example repeatable. Time is in arbitrary units."
        ),
        nbformat.v4.new_markdown_cell("## 1. Generate observed events and censoring"),
        nbformat.v4.new_code_cell(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from kesurv import KESurv\n"
            "from sksurv.ensemble import RandomSurvivalForest\n"
            "from sksurv.nonparametric import kaplan_meier_estimator\n"
            "from sksurv.tree import SurvivalTree\n"
            "from sksurv.util import Surv\n\n"
            "rng = np.random.default_rng(2026)\n"
            "n_subjects = 320\n"
            "X = rng.normal(size=(n_subjects, 5))\n"
            "latent_risk = 0.85 * X[:, 0] - 0.65 * X[:, 1] + 0.25 * X[:, 2]\n"
            "event_time = rng.exponential(scale=2.5 * np.exp(-latent_risk))\n"
            "censor_time = rng.exponential(scale=4.0, size=n_subjects)\n"
            "event = event_time <= censor_time\n"
            "observed_time = np.minimum(event_time, censor_time)\n"
            "y = Surv.from_arrays(event, observed_time)\n"
            "X_train, X_test = X[:250], X[250:]\n"
            "y_train, y_test = y[:250], y[250:]\n"
            "risk_test = latent_risk[250:]  # only for choosing examples to display\n"
            "print(f'Training subjects: {len(X_train)}')\n"
            "print(f'Observed event fraction: {y_train[y_train.dtype.names[0]].mean():.1%}')"
        ),
        nbformat.v4.new_markdown_cell(
            "## 2. Fit the ensemble\n\n"
            "The forest supplies local candidates. Two survival trees vote on whose "
            "predicted trajectories look similar to each query subject."
        ),
        nbformat.v4.new_code_cell(
            "model = KESurv(\n"
            "    base_estimators=[\n"
            "        ('shallow', SurvivalTree(max_depth=3, random_state=11)),\n"
            "        ('deep', SurvivalTree(max_depth=6, random_state=12)),\n"
            "    ],\n"
            "    forest=RandomSurvivalForest(\n"
            "        n_estimators=60, min_samples_leaf=5, random_state=13\n"
            "    ),\n"
            "    epsilon=0.08,\n"
            "    min_agreement=1,\n"
            "    similarity_time=3.0,\n"
            "    n_timepoints=64,\n"
            "    backend='numpy',\n"
            "    random_state=2026,\n"
            ").fit(X_train, y_train)\n"
            "print(f'Similarity grid: {len(model.similarity_times_)} points')\n"
            "print(f'Event times in output: {len(model.unique_times_)}')"
        ),
        nbformat.v4.new_markdown_cell(
            "## 3. See the survival curves\n\n"
            "We select three held-out subjects using the **known synthetic latent risk** "
            "solely to make the plot easy to read. That risk is not supplied to KESurv. "
            "The dashed population curve ignores subject features."
        ),
        nbformat.v4.new_code_cell(
            "order = np.argsort(risk_test)\n"
            "chosen = np.array([order[5], order[len(order) // 2], order[-6]])\n"
            "horizon = float(np.quantile(y_train[y_train.dtype.names[1]], 0.95))\n"
            "times = np.linspace(0.0, horizon, 180)\n"
            "survival = model.predict_survival(X_test[chosen], times=times)\n"
            "km_times, km_survival = kaplan_meier_estimator(\n"
            "    y_train[y_train.dtype.names[0]], y_train[y_train.dtype.names[1]]\n"
            ")\n\n"
            "fig, ax = plt.subplots(figsize=(9, 5.6), dpi=125)\n"
            "shades = ['#111111', '#5c5c5c', '#a3a3a3']\n"
            "labels = ['Lower synthetic risk', 'Middle synthetic risk', 'Higher synthetic risk']\n"
            "for row, (color, label) in enumerate(zip(shades, labels)):\n"
            "    ax.step(times, survival[row], where='post', lw=2.2, color=color, label=label)\n"
            "ax.step(km_times, km_survival, where='post', lw=1.4, ls=(0, (4, 3)),\n"
            "        color='#111111', label='Population Kaplan–Meier')\n"
            "ax.set(xlim=(0, horizon), ylim=(0, 1.02), xlabel='Time (arbitrary units)',\n"
            "       ylabel='Estimated survival probability',\n"
            "       title='KESurv: patient-specific survival on synthetic data')\n"
            "ax.spines[['top', 'right']].set_visible(False)\n"
            "ax.grid(axis='y', color='#e6e6e6')\n"
            "ax.legend(frameon=False, loc='upper center', ncol=2,\n"
            "          bbox_to_anchor=(0.5, -0.16))\n"
            "fig.tight_layout()\n"
            "plt.show()"
        ),
        nbformat.v4.new_markdown_cell(
            "## 4. Inspect one prediction\n\n"
            "KESurv exposes selected neighbors, their weights, learner votes, and any "
            "fallback to the forest distribution."
        ),
        nbformat.v4.new_code_cell(
            "info = model.predict_neighbor_info(X_test[chosen[-1:]])[0]\n"
            "print(f'Forest candidates: {len(info[\"candidate_indices\"])}')\n"
            "print(f'Selected neighbors: {len(info[\"selected_indices\"])}')\n"
            "print(f'Fallback used: {info[\"fallback\"]}')\n"
            "print(f'Weight sum: {info[\"selected_weights\"].sum():.3f}')\n"
            "print(f'RMST through time 3: {model.predict_rmst(X_test[chosen[-1:]], 3.0)[0]:.3f}')"
        ),
        nbformat.v4.new_markdown_cell(
            "These results illustrate API behavior on generated data. They are not a "
            "clinical validation or a reproduction of the paper's benchmark tables. "
            "To rerun the notebook after changing the package, use "
            "`python examples/build_synthetic_notebook.py`."
        ),
    ]

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )
    # The Jupyter kernelspec uses `python`; put the active virtual environment first.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    runtime_dir = ROOT / ".tmp" / "jupyter_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime_dir)
    os.environ["IPYTHONDIR"] = str(ROOT / ".tmp" / "ipython")
    NotebookClient(notebook, timeout=180, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    images = [
        output.data["image/png"]
        for cell in notebook.cells if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    if len(images) != 1:
        raise RuntimeError(f"expected one survival-curve plot, found {len(images)}")
    (ROOT / "docs" / "_static" / "synthetic-survival-curves.png").write_bytes(
        base64.b64decode(images[0])
    )
    nbformat.write(notebook, OUTPUT)
    print(f"Wrote executed notebook to {OUTPUT}")


if __name__ == "__main__":
    main()
