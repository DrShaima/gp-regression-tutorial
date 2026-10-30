"""Run trusted tutorial code cells without Jupyter. Author: Sheima Alameeri."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
notebook = json.loads((root / "prediction_intervals.ipynb").read_text())
namespace = {"__name__": "__main__"}
figure_number = 0
def save_figures():
    global figure_number
    for number in plt.get_fignums():
        figure_number += 1
        plt.figure(number).savefig(root / f"figure_{figure_number}.png", dpi=150, bbox_inches="tight")
    plt.close("all")
plt.show = save_figures
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        exec(compile("".join(source) if isinstance(source, list) else source, "prediction_intervals.ipynb", "exec"), namespace)
print("Finished; figures saved beside this script.")
