import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

TARGET_PERCENT = 90
N_CALIBRATION = 199
REGION_EDGES = np.linspace(-1, 1, 5)


def design(x):
    return np.column_stack([np.ones(x.size), x])


def sample(rng, n):
    x = rng.uniform(-1, 1, n)
    noise_sd = 0.4 * np.exp(x)
    y = 0.5 + x + noise_sd * rng.normal(size=n)
    return x, y


def calibration_threshold(scores, target_percent):
    # Integer ceiling keeps exact ranks for whole-number coverage percentages.
    rank = ((scores.size + 1) * target_percent + 99) // 100
    if rank > scores.size:
        return np.inf
    return np.sort(scores)[rank - 1]


def fit_and_calibrate(rng):
    x_fit, y_fit = sample(rng, 160)
    mean_coef = np.linalg.lstsq(design(x_fit), y_fit, rcond=None)[0]

    x_scale, y_scale = sample(rng, 240)
    abs_error = np.abs(y_scale - design(x_scale) @ mean_coef)
    log_error = np.log(np.maximum(abs_error, 1e-6))
    scale_coef = np.linalg.lstsq(design(x_scale), log_error, rcond=None)[0]

    x_cal, y_cal = sample(rng, N_CALIBRATION)
    cal_error = np.abs(y_cal - design(x_cal) @ mean_coef)
    cal_scale = np.exp(design(x_cal) @ scale_coef)
    constant_q = calibration_threshold(cal_error, TARGET_PERCENT)
    scaled_q = calibration_threshold(cal_error / cal_scale, TARGET_PERCENT)
    return mean_coef, scale_coef, constant_q, scaled_q


def predict(x, fit):
    mean_coef, scale_coef, constant_q, scaled_q = fit
    center = design(x) @ mean_coef
    constant_radius = np.full(x.size, constant_q)
    adaptive_radius = scaled_q * np.exp(design(x) @ scale_coef)
    return center, np.vstack([constant_radius, adaptive_radius])


def main():
    # Independent random streams for the example plot and repeated experiment.
    plot_seed, experiment_seed = np.random.SeedSequence(81025).spawn(2)
    plot_rng = np.random.default_rng(plot_seed)
    fit = fit_and_calibrate(plot_rng)
    x_plot, y_plot = sample(plot_rng, 250)
    grid = np.linspace(-1, 1, 300)
    center, radii = predict(grid, fit)

    rng = np.random.default_rng(experiment_seed)
    repetitions, n_test = 400, 1500
    hits = np.zeros((2, 4), dtype=int)
    widths = np.zeros((2, 4))
    counts = np.zeros(4, dtype=int)
    for _ in range(repetitions):
        fitted = fit_and_calibrate(rng)
        x, y = sample(rng, n_test)
        test_center, test_radii = predict(x, fitted)
        covered = np.abs(y - test_center) <= test_radii
        for j, (left, right) in enumerate(zip(REGION_EDGES[:-1], REGION_EDGES[1:])):
            in_region = (left <= x) & (x < right)
            counts[j] += in_region.sum()
            hits[:, j] += covered[:, in_region].sum(axis=1)
            widths[:, j] += (2 * test_radii[:, in_region]).sum(axis=1)

    coverage_by_region = hits / counts
    width_by_region = widths / counts
    overall_coverage = hits.sum(axis=1) / counts.sum()
    overall_width = widths.sum(axis=1) / counts.sum()

    labels = ["Constant width", "Adaptive width"]
    print(f"{repetitions} independent runs with {n_test} test points each")
    print(f"{'Method':<16}{'Overall coverage':>18}{'Mean width':>13}")
    for label, coverage, width in zip(labels, overall_coverage, overall_width):
        print(f"{label:<16}{coverage:>18.2%}{width:>13.3f}")
    print()
    print(f"{'Input region':<16}{'Constant cov.':>15}{'Adaptive cov.':>15}"
          f"{'Constant width':>17}{'Adaptive width':>17}")
    region_labels = ["[-1, -0.5)", "[-0.5, 0)", "[0, 0.5)", "[0.5, 1)"]
    for j, region in enumerate(region_labels):
        print(f"{region:<16}{coverage_by_region[0, j]:>15.2%}"
              f"{coverage_by_region[1, j]:>15.2%}{width_by_region[0, j]:>17.3f}"
              f"{width_by_region[1, j]:>17.3f}")

    blue, orange = "#286996", "#c45717"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    ax = axes[0]
    ax.scatter(x_plot, y_plot, color="#374151", s=10, alpha=0.25, label="Test responses")
    ax.plot(grid, center, color="#111827", linewidth=1.6, label="Fitted mean")
    ax.plot(grid, center + radii[0], "--", color=blue, label=labels[0])
    ax.plot(grid, center - radii[0], "--", color=blue)
    ax.plot(grid, center + radii[1], color=orange, label=labels[1])
    ax.plot(grid, center - radii[1], color=orange)
    ax.set(title="Prediction intervals for one split", xlabel="Input x", ylabel="Response")
    ax.legend(fontsize=8)

    ax = axes[1]
    positions = np.arange(len(region_labels))
    ax.bar(positions - 0.18, coverage_by_region[0], 0.36, color=blue, label=labels[0])
    ax.bar(positions + 0.18, coverage_by_region[1], 0.36, color=orange, label=labels[1])
    ax.axhline(TARGET_PERCENT / 100, color="#111827", linestyle="--", linewidth=1,
               label="Marginal target")
    ax.set(xticks=positions, xticklabels=region_labels, ylim=(0, 1.08),
           title="Coverage by input region across 400 runs", xlabel="Input region",
           ylabel="Observed coverage")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.legend(loc="lower left", fontsize=8)
    plt.show()


if __name__ == "__main__":
    main()
