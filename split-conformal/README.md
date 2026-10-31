# Split conformal prediction: calibrating prediction intervals

Continue from [Prediction intervals and coverage](../prediction-intervals/README.md).

## Learning goals

Build a prediction interval using held-out residuals, understand the finite-sample rank correction, and distinguish marginal coverage from coverage in specific input regions. This standalone educational example uses synthetic data and established methods; it contains no dissertation code.

## 1. Keep three roles separate

**Training** fits the predictor. **Calibration** sets the interval width. **Test** evaluates the final procedure. Choose the model and error level before examining calibration or test outcomes. Hyperparameter selection belongs inside the training portion. Refitting on calibration data after calculating the threshold invalidates this construction.

We deliberately fit a straight line to a nonlinear mean. Conformal prediction does not require a correct regression model for marginal validity under the assumptions below, but a poor predictor can produce unnecessarily wide intervals.

## 2. Turn prediction errors into scores

Let $\widehat f$ be the predictor fitted using training data only. For calibration pair $(X_i,Y_i)$, define

$$
R_i=|Y_i-\widehat f(X_i)|,\qquad i=1,\ldots,m.
$$

Here $m$ is the calibration sample size. A larger score means a larger prediction error. Sort these scores as $R_{(1)}\le\cdots\le R_{(m)}$.

For target miscoverage $\alpha$ (for example, $\alpha=0.10$ for 90% coverage), set

$$
k=\left\lceil(m+1)(1-\alpha)\right\rceil.
$$

The ceiling rounds upward. Use the actual order statistic, without interpolation:

$$
\widehat q=\begin{cases}R_{(k)},&k\le m,\\+\infty,&k=m+1.\end{cases}
$$

**Worked example:** with $m=199$ and $\alpha=0.10$, $k=180$. Use the 180th smallest absolute residual, which is element 179 in a zero-indexed sorted Python array. With $m=8$ and $\alpha=0.10$, $k=9$ and the valid interval is unbounded. Do not silently replace it with the largest available residual.

## 3. Construct a future-response interval

$$
C(x)=[\widehat f(x)-\widehat q,\;\widehat f(x)+\widehat q].
$$

Its width is $2\widehat q$ everywhere. This interval targets one future observed response, including observation noise. It is not an interval for the unknown mean function.

## 4. What is guaranteed?

Conditional on the training data, assume the calibration pairs and the future pair are exchangeable: their joint distribution is unchanged by permutation. Independent observations from the same distribution satisfy this assumption. With the predictor fixed using training data only,

$$
\Pr\{Y_{\mathrm{new}}\in C(X_{\mathrm{new}})\}\ge 1-\alpha.
$$

The probability averages over calibration data and the future pair (and may also average over training data). It is not a claim of 90% coverage for each realized calibration set, each fixed input, or every subgroup. It also does not guarantee simultaneous coverage of all future observations. Arbitrary distribution shifts or dependent time series need additional analysis.

**Why the correction works:** when scores have no ties, the future score's rank among $m+1$ exchangeable scores is uniform. The first $k$ ranks are covered, giving $k/(m+1)\ge1-\alpha$. Ties can make the procedure conservative. The unbounded case covers every finite response.

## 5. Experiment

Generate independent pairs with $X$ uniform on $[-1,1]$, standard-normal $\varepsilon$ independent of $X$, and

$$
Y=\sin(\pi X)+\sigma(X)\varepsilon,\qquad \sigma(x)=0.15+0.65\frac{x+1}{2}.
$$

Noise grows from left to right. Fit an intercept and slope to 120 training pairs, calibrate on 199 fresh pairs, and evaluate on 5,000 independent test pairs. Both model misspecification and changing noise are intentional.

Test coverage and average width are

$$
\widehat c=\frac{1}{N}\sum_{j=1}^{N}\mathbf 1\{L_j\le Y_j\le U_j\},\qquad
\overline w=\frac{1}{N}\sum_{j=1}^{N}(U_j-L_j).
$$

Here $N$ is test size; $L_j,U_j$ are interval endpoints. The indicator equals one for a captured response. Repeat the entire train/calibrate/test procedure 500 times, using one fresh future pair per repetition. These independent success indicators estimate marginal coverage; their Monte Carlo standard error is $\sqrt{\widehat c(1-\widehat c)/500}$. Test points sharing one calibration set are not independent replications of the full procedure.

## Run the example

Open [the executed notebook](split_conformal.ipynb) for saved plots and results. From the repository root:

```bash
python -m pip install -r requirements.txt
python split-conformal/run_example.py
```

The script displays plots and prints metrics; it does not update notebook outputs. For interactive use, install JupyterLab separately and run all notebook cells.

## Saved output

```text
Single split: coverage=0.9008, width=2.1676
Region [-1.0, -0.5): n=1217, coverage=0.9729
Region [-0.5, 0.0): n=1220, coverage=0.9648
Region [0.0, 0.5): n=1279, coverage=0.8663
Region [0.5, 1.0): n=1284, coverage=0.8061
Independent repetitions: B=500, coverage=0.9200, MCSE=0.0121
95% exact binomial interval: [0.8927, 0.9422]
Average width across repetitions: 2.1926
Rank, boundary, tie, input-validation, and finite-width checks passed.
```

## 6. Read the results carefully

The first plot shows one fitted/calibrated interval. Its width stays constant even where noise changes. Regional coverage in the second plot can vary substantially; these values describe this one split and are not population guarantees. The repeated experiment averages over new training and calibration sets. Its exact binomial interval describes simulation uncertainty in the procedure's marginal coverage, not uncertainty about a future response.

Simulation illustrates the theorem; it cannot prove it. A reported coverage below 0.90 in a finite run can occur from Monte Carlo variation. Better point prediction can improve efficiency, but conformal calibration does not remove bias or make the mean estimate correct.

## 7. Exercises

1. Change alpha to 0.05 in a fresh run. Explain the rank change and compare widths.
2. Increase calibration size. Which source of randomness does this reduce?
3. Replace the line by a cubic polynomial fitted only on training data. Compare widths over independent repetitions, keeping the test data out of model selection.
4. Set noise constant. Does constant interval width become more appropriate?

A future extension is a positive, training-fitted scale function: divide absolute residuals by that scale before calibration, then multiply the threshold by the scale at prediction time. This can adapt interval widths, but does not automatically give conditional coverage.

## References and reproducibility

- Oliveira, R. I., Orenstein, P., Ramos, T., and Romano, J. V. (2024). [Split Conformal Prediction and Non-Exchangeable Data](https://jmlr.org/papers/v25/23-1553.html). *Journal of Machine Learning Research*, 25, 1–38. Background and extensions; this tutorial uses the exchangeable setting with the explicit finite-sample rank correction.
- [SciPy exact binomial confidence intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).

Saved execution: Python 3.11+, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8. The repository requirements supply these dependencies. Fixed seeds reproduce this experiment in that environment; they do not establish validity for other data-generating processes. Newly written educational code and exposition prepared with AI assistance.
