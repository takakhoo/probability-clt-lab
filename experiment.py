"""Exact binomial interval coverage, estimator risk, and a seeded CLT check."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import binom, norm


def intervals(k, n, method="wilson", confidence=0.95):
    k = np.asarray(k, dtype=float)
    if not isinstance(n, (int, np.integer)) or n < 1 or not 0 < confidence < 1:
        raise ValueError("Positive integer n and confidence in (0,1) required")
    if not np.all(np.isfinite(k)) or np.any((k < 0) | (k > n) | (k != np.floor(k))):
        raise ValueError("Success counts must be integers in [0,n]")
    z = norm.ppf((1 + confidence) / 2)
    p = k / n
    if method == "wald":
        half = z * np.sqrt(p * (1 - p) / n)
        return np.maximum(0, p - half), np.minimum(1, p + half)
    if method != "wilson":
        raise ValueError("method must be wald or wilson")
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    # Floating-point rounding must not exclude the endpoints p=0 or p=1.
    return np.where(k == 0, 0, center - half), np.where(k == n, 1, center + half)


def exact_coverage(n, p, method):
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0,1]")
    k = np.arange(n + 1)
    low, high = intervals(k, n, method)
    return float(binom.pmf(k, n, p)[(low <= p) & (p <= high)].sum())


def estimator_risks(n, p, a=6, b=10):
    if n < 1 or int(n) != n or not 0 <= p <= 1 or min(a, b) <= 0:
        raise ValueError("Invalid n, p, or Beta prior")
    variance = p * (1 - p)
    mle = variance / n
    # Frequentist MSE of the posterior mean, not posterior uncertainty.
    posterior_mean = (n * variance + (a - (a + b) * p) ** 2) / (n + a + b) ** 2
    return float(mle), float(posterior_mean)


def run(output="results", seed=7, repetitions=100000):
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    for p in [0.01, 0.1, 0.5, 0.9]:
        for n in [10, 50, 200, 1000]:
            counts = rng.binomial(n, p, size=repetitions)
            mle, bayes = estimator_risks(n, p)
            rows.append(dict(n=n, p=p, wald_coverage=exact_coverage(n, p, "wald"),
                             wilson_coverage=exact_coverage(n, p, "wilson"),
                             mle_mse_exact=mle, beta_6_10_mse_exact=bayes,
                             mle_mse_simulated=float(np.mean((counts / n - p) ** 2)),
                             beta_6_10_mse_simulated=float(np.mean(((counts + 6) / (n + 16) - p) ** 2))))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    result = dict(seed=seed, repetitions=repetitions, confidence=0.95, prior=[6, 10], rows=rows)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    grid = np.linspace(0.001, 0.999, 999)
    for method, color in [("wald", "#b84a45"), ("wilson", "#227e96")]:
        axes[0].plot(grid, [exact_coverage(50, p, method) for p in grid], label=method.title(), color=color)
    axes[0].axhline(0.95, color="gray", ls="--", lw=1)
    axes[0].set(title="95% intervals: exact coverage, n=50", xlabel="True probability p", ylabel="Coverage", ylim=(0, 1.02))
    axes[0].legend(frameon=False)
    risks = np.array([estimator_risks(10, p) for p in grid])
    axes[1].plot(grid, risks[:, 0], label="MLE")
    axes[1].plot(grid, risks[:, 1], label="Beta(6,10) posterior mean")
    axes[1].set(title="A strong prior helps only sometimes", xlabel="True probability p", ylabel="Exact MSE, n=10")
    axes[1].legend(frameon=False, fontsize=8)
    for p, n, color in [(0.01, 50, "#b84a45"), (0.5, 200, "#227e96")]:
        k = np.arange(n + 1)
        z = (k - n * p) / np.sqrt(n * p * (1 - p))
        axes[2].step(z, binom.cdf(k, n, p), where="post", label=f"p={p}, n={n}", color=color)
    z = np.linspace(-4, 4, 400)
    axes[2].plot(z, norm.cdf(z), "k--", label="Standard normal")
    axes[2].set(title="CLT: rare events converge slowly", xlabel="Standardized count", ylabel="CDF", xlim=(-3, 3))
    axes[2].legend(frameon=False, fontsize=8)
    fig.savefig(out / "probability-lab.png", dpi=160)
    plt.close(fig)
    print(json.dumps(rows[:4], indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args.output, args.seed)
