# Prediction intervals and coverage

**Sheima Alameeri, PhD**

Open [the executed notebook](prediction_intervals.ipynb) for readable mathematical equations, explanations of every symbol, saved graphs, and runnable Python. This follows the [main GP regression tutorial](../README.md).

The example compares latent-function and future-observation intervals, measures held-out coverage and width, and runs 150 repeated simulations to show the effect of underestimating noise. It uses synthetic data and is separate from dissertation code.

## Run

From the repository root, using Python 3.11 or newer:

```bash
python -m pip install -r requirements.txt
python prediction-intervals/run_example.py
```

The script runs this trusted notebook's code cells and saves two figures in this folder; it does not refresh notebook outputs. For interactive editing, install JupyterLab separately and open the notebook.

Saved execution used NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8. No new dependencies beyond the repository requirements are needed.

Intervals are pointwise and conditional on fixed hyperparameters. The simulation uses a fixed function, independent noise across repetitions, and no extrapolation. It is an educational illustration, not a universal calibration guarantee.
