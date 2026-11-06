# Adaptive-width conformal prediction

The [split conformal tutorial](../split-conformal/README.md) used one interval width for every input. Here we fit a second model that estimates where prediction errors tend to be larger. We compare both methods using the same point predictor, calibration observations, and test responses.

## 1. Give each input its own scale

Let $\widehat f(x)$ predict the response at input $x$. Let $\widehat s(x)>0$ estimate the typical size of a prediction error there. A scale can describe the *relative* size of errors across inputs. Its units or overall multiplier are absorbed by calibration.

We fit $\widehat f$ on the first training subset. On a second training subset, we fit a straight line to the log absolute residuals. For fitted coefficients $\widehat a,\widehat b$, the scale is

```math
\widehat s(x)=\exp(\widehat a+\widehat b x).
```

The exponential keeps the scale positive. Before taking logs, the code floors residual magnitudes at $10^{-6}$. This fitted scale summarizes residual size and may reflect both noise and mean prediction error. Both fits are frozen before calibration.

## 2. Calibrate the normalized residuals

On the calibration set, compute an absolute error $R_i$ and a normalized score $S_i$ for each pair $(X_i,Y_i)$.

```math
R_i=\bigl|Y_i-\widehat f(X_i)\bigr|,
\qquad
S_i=\frac{R_i}{\widehat s(X_i)},
\qquad i=1,\dots,m.
```

For $m$ calibration observations and target coverage $1-\alpha$, use the rank

```math
k=\left\lceil(m+1)(1-\alpha)\right\rceil.
```

The ceiling rounds up. Let $q_R=R_{(k)}$ and $q_S=S_{(k)}$, where $(k)$ denotes the $k$th smallest value in each respective score list. When $k=m+1$, set both thresholds to $+\infty$. For $m=199$ and 90% coverage, use rank $180$.

## 3. Convert back to response units

The constant and adaptive prediction intervals are

```math
C_{\mathrm{constant}}(x)
=\left[\widehat f(x)-q_R,\ \widehat f(x)+q_R\right],
```

```math
C_{\mathrm{adaptive}}(x)
=\left[\widehat f(x)-q_S\widehat s(x),\
       \widehat f(x)+q_S\widehat s(x)\right].
```

The normalized score divides by the scale. The interval radius multiplies by it. For example, with $\widehat f(x)=3$, $\widehat s(x)=0.4$, and $q_S=1.5$, the radius is $0.6$ and the interval is $[2.4,3.6]$.

Multiplying every fitted scale by a constant $c>0$ divides every normalized calibration score and $q_S$ by $c$. The final radius stays the same.

## 4. Interpret the guarantee

Assume the calibration observations and one future pair are exchangeable conditional on the two training subsets. Independent observations from the same distribution meet this assumption. The same frozen score function applies to every calibration observation and to the future pair, so the scores are exchangeable as well. With no ties, the rank of the future score among the $m+1$ scores is uniform. With ties, including scores equal to the threshold preserves the lower bound.

For either interval rule $C$, the guarantee is

```math
\Pr\!\left\{Y_{\mathrm{new}}\in C(X_{\mathrm{new}})\right\}
\ge 1-\alpha.
```

The probability averages over the calibration data and the future pair, with the fitted models held fixed. It is a marginal guarantee for one future response. Coverage within a fixed input region can differ from the target even when this guarantee holds.

## 5. Compare both methods on the same data

Generate $X$ uniformly on $[-1,1]$ and an independent standard normal error $\varepsilon$. Set

```math
Y=0.5+X+0.4\exp(X)\,\varepsilon.
```

The noise standard deviation increases with $X$. Each of the 400 runs generates 160 observations to fit the mean, 240 to fit the scale, 199 for calibration, and 1,500 for evaluation. For a region $A$, observed coverage is

```math
\widehat c_A
=\frac{\sum_j \mathbf{1}\!\left\{X_j\in A,\ Y_j\in C_j(X_j)\right\}}
       {\sum_j \mathbf{1}\!\left\{X_j\in A\right\}}.
```

Here $j$ indexes the pooled test responses and $C_j$ is the interval from the run that produced test response $j$. Mean width averages $U_j-L_j$ over those same test inputs, where $L_j$ and $U_j$ are the lower and upper endpoints. Test responses within a run share the fitted models and calibration data. Treat the 400 runs as the independent units if you want to quantify simulation uncertainty.

## Run the example

Open the [executed notebook](adaptive_conformal.ipynb) for the code, plots, and saved results. To rerun the example, use Python 3.11 or newer from the repository root.

```bash
python -m pip install -r requirements.txt
python adaptive-conformal/run_example.py
```

The script uses synthetic data and fixed random seeds. It prints the results and displays the plots.

## Saved results

```text
400 independent runs with 1500 test points each
Method            Overall coverage   Mean width
Constant width              89.84%        1.786
Adaptive width              90.02%        1.579

Input region      Constant cov.  Adaptive cov.   Constant width   Adaptive width
[-1, -0.5)               99.99%         89.92%            1.786            0.651
[-0.5, 0)                99.04%         90.41%            1.786            1.062
[0, 0.5)                 90.29%         90.05%            1.785            1.740
[0.5, 1)                 70.08%         89.70%            1.786            2.860
```

## 6. What changed?

In this simulation, both methods gave close to 90% overall coverage. With constant width, coverage fell to about 70% in the highest noise region. The adaptive method brought all four regional estimates within about half a percentage point of 90%. It used narrower intervals in quiet regions and wider ones in noisy regions. Its mean width was about 12% smaller in this example.

The scale model can help when it captures how errors vary across inputs. The regional estimates and width savings are specific to this simulation. A different noise pattern or a poor scale fit can change both results.

Try replacing the increasing noise scale in the generator with a constant, then compare regional coverage and mean width again. Next try a U-shaped noise scale and see whether a log-linear scale model is flexible enough.

## Further reading

- Lei, G’Sell, Rinaldo, Tibshirani, and Wasserman (2018), *Distribution-Free Predictive Inference for Regression*. [Journal of the American Statistical Association](https://doi.org/10.1080/01621459.2017.1307116).
- Oliveira, Orenstein, Ramos, and Romano (2024), *Split Conformal Prediction and Non-Exchangeable Data*. [Journal of Machine Learning Research](https://jmlr.org/papers/v25/23-1553.html).
