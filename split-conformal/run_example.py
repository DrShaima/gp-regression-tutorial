import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binomtest

ALPHA = 0.10

def threshold(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Supply a nonempty one-dimensional score array")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("Scores must be finite and nonnegative")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return np.inf if k > len(scores) else np.sort(scores)[k - 1]

def sample(rng, n):
    x = rng.uniform(-1, 1, n)
    y = np.sin(np.pi * x) + (0.15 + 0.65 * (x + 1) / 2) * rng.normal(size=n)
    return x, y

def design(x):
    return np.column_stack([np.ones(len(x)), x])

def fit_and_calibrate(rng):
    xt, yt = sample(rng, 120)
    xc, yc = sample(rng, 199)
    beta = np.linalg.lstsq(design(xt), yt, rcond=None)[0]
    q = threshold(np.abs(yc - design(xc) @ beta), ALPHA)
    return beta, q

# Check exact rank, ties, and the small-calibration unbounded case.
assert threshold(np.arange(1, 200), 0.10) == 180
assert np.isinf(threshold(np.arange(8), 0.10))
assert threshold(np.ones(199), 0.10) == 1
for bad_scores, bad_alpha in [([], .1), ([np.nan], .1), ([-1], .1), ([1], 0)]:
    try:
        threshold(bad_scores, bad_alpha)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid input was accepted")

rng = np.random.default_rng(20261031)
beta, q = fit_and_calibrate(rng)
x, y = sample(rng, 5000)
mu = design(x) @ beta
covered = np.abs(y - mu) <= q
print(f"Single split: coverage={covered.mean():.4f}, width={2*q:.4f}")
edges = np.linspace(-1, 1, 5)
local = []
for left, right in zip(edges[:-1], edges[1:]):
    mask = (x >= left) & (x < right)
    local.append(covered[mask].mean())
    print(f"Region [{left:.1f}, {right:.1f}): n={mask.sum()}, coverage={local[-1]:.4f}")

grid = np.linspace(-1, 1, 300)
line = design(grid) @ beta
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
axes[0].scatter(x[:350], y[:350], s=10, alpha=.35, label="Fresh test responses (subset)")
axes[0].fill_between(grid, line-q, line+q, alpha=.22, label="90% split conformal interval")
axes[0].plot(grid, line, label="Fitted line")
axes[0].plot(grid, np.sin(np.pi*grid), "--", label="True mean")
axes[0].set(xlabel="Input x", ylabel="Response", title="One fitted/calibrated interval")
axes[0].legend(fontsize=8)
axes[1].bar(["[-1,-0.5)", "[-0.5,0)", "[0,0.5)", "[0.5,1)"], local, color="#327eaa")
axes[1].axhline(.9, color="black", linestyle="--", label="Marginal target")
axes[1].set(ylim=(0,1.05), xlabel="Input region", ylabel="Observed coverage", title="Regional coverage can differ")
axes[1].tick_params(axis="x", labelsize=8)
axes[1].legend(fontsize=8)
plt.show()

# Independent datasets and one independent future pair for each repetition.
rep_rng = np.random.default_rng(20261101)
B = 500
hits = np.empty(B, dtype=bool)
widths = np.empty(B)
for b in range(B):
    beta_b, q_b = fit_and_calibrate(rep_rng)
    xb, yb = sample(rep_rng, 1)
    hits[b] = abs(yb[0] - (design(xb) @ beta_b)[0]) <= q_b
    widths[b] = 2*q_b
coverage = hits.mean()
mcse = np.sqrt(coverage*(1-coverage)/B)
ci = binomtest(int(hits.sum()), B).proportion_ci(method="exact")
print(f"Independent repetitions: B={B}, coverage={coverage:.4f}, MCSE={mcse:.4f}")
print(f"95% exact binomial interval: [{ci.low:.4f}, {ci.high:.4f}]")
print(f"Average width across repetitions: {widths.mean():.4f}")
assert np.isfinite(widths).all() and (widths >= 0).all()
print("Rank, boundary, tie, input-validation, and finite-width checks passed.")
