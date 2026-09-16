# Probability Experiments: Bernoulli Estimation and the Central Limit Theorem

A compact, executable study of discrete and continuous random variables,
Bernoulli parameter estimation, and the Central Limit Theorem (CLT). The
notebook connects each mathematical result to a NumPy/SciPy experiment and a
visual check.

## What this demonstrates

- Expected value and variance for a discrete distribution
- Numerical integration for a continuous probability density
- Maximum-likelihood and Bayesian estimates for a Bernoulli parameter
- Convergence of sample means toward a Gaussian distribution as sample size
  increases

## Quick start

```bash
git clone https://github.com/takakhoo/probability-clt-lab.git
cd probability-clt-lab
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab "Bernoulli CLT Experiment.ipynb"
```

[Open the executed notebook](Bernoulli%20CLT%20Experiment.ipynb)

## Mathematical focus

For independent Bernoulli trials with success probability \(p\), the notebook
compares the empirical sampling distribution with

\[
\sqrt{n}(\bar X_n-p) \xrightarrow{d} \mathcal{N}(0,p(1-p)).
\]

The cells are arranged as an exploratory lab: change the prior, success
probability, sample size, or number of simulations and rerun from top to bottom.

## Verification

The committed notebook was executed from its first cell through its last in a
fresh environment on September 16, 2026. It completed without cell errors and
regenerated the estimators and CLT visualizations.

## Scope

This is an educational experiment, not a general-purpose statistics package.
It is useful as a transparent reference for the assumptions and mechanics
behind common estimators and asymptotic approximations.
