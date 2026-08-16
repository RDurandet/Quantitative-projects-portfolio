## Project Summary
This project simulates a high-energy physics lifetime-measurement experiment using a Monte Carlo model of particle decays and a segmented detector array. Gaussian detector error is added to simulate detector mismeasurement. The pipeline then reconstructs decay vertices using recorded detector hits, accounts for geometric biases, and fits an analytically-derived geometric probability density function (PDF) to recover the mean lifetime. The analysis recovers the mean lifetime within $0.96\sigma$ of the true input, with uncertainties propagated explicitly and key steps cross‑checked against independent analytic calculations.

## Environment
- Python

## Motivation and problem
**Goal**: measure the mean lifetime of an unstable particle from detector data in a realistic geometry with finite acceptance and detector smearing. <br>
**Challenges addressed**: biased detector acceptance, sparse hits per particle, regression uncertainty, and geometric normalisation of reconstructed decay positions.

## Simulation and numerical methods
- Generated $N$ statistically independent events with a single event‑generation function, enabling trivial parallelisation across CPU cores for large $N$.
- Samples daughter‑particle directions uniformly over solid angle via sphere‑point‑picking (uniform in azimuth $\phi$ and $\cos\theta$), ensuring unbiased angular coverage.
- Models detector hits by propagating trajectories through a stack of planar detectors, with early‑exit logic to skip geometrically impossible hits and miss‑after‑first‑miss paths.
- Uses seeded random number generation to guarantee full reproducibility of velocity, lifetime, direction, and smearing draws across runs.

## Performance and parallelisation
- Runs embarrassingly parallel event generation with `ipyparallel` in Jupyter, distributing batches of events across multiple cores.
- Optimised data transfer by only serialising hit‑related coordinates; avoids pickling $\sim1.2\text{GB}$ of redundant data for $10^7$ no-hit particles.

## Detector Modelling and data handling
- Simulates a four‑detector stack; front detector sees hits from $\sim 4.5\%$ of particles, with spatial histograms confirming uniform face coverage and central‑region solid‑angle enhancement.
-  Implements conditional storage of smear/unsmeared coordinates only for hit events, reducing total detector‑calculation load by $\sim 87\%$ compared to a naive algorithm.
-  Returns per-event results as dictionaries, aggregated into `pandas` DataFrames for analysis and plotting.

## Vertex reconstruction and lifetime fitting
- Reconstructs particle tracks via linear regression on detector hit positions, keeping only tracks whose fitted vertex lies within $0.1$m of the beam axis to control regression error.
- Restricts reconstruction to particles that hit all detectors, trading sample size for improved fit stability and reduced bias.
- Derives the decay‑position PDF by convolving a Gaussian velocity distribution with an exponential lifetime distribution and fits this to the geometrically normalised histogram.
- Evaluates the convolution integral using adaptive quadrature (`scipy.integrate.quad`), chosen for robust error control and efficiency on smooth integrands.
- Histogram binning is chosen to balance geometric normalisation and statistical uncertainty (e.g. 250 bins in decay‑position space), and the lifetime fit is evaluated under this trade‑off.

## Key results
- Lifetime estimate: $\tau_\text{fit} = 2.52 \pm 0.02$ms vs true $\tau = 2.50$ms, corresponding to a $0.96\sigma$ deviation.
- Detector-hit savings: $\sim 5.1$ million actual detector checks vs a naive $40$ million, yielding an $87.25\%$ reduction in geometric computations.
- Hit‑multiplicity distribution shows that 1‑ and 4‑detector hits dominate, with 2–3 detector hits confined to narrow angular windows imposed by the geometry.

## Limitations and future work
- Current reconstruction assumes perfect track association; realistic high‑multiplicity environments would require explicit combinatorial track‑finding and ambiguity resolution.
- Detector smearing is treated with a simple model; full unfolding (e.g. iterative Bayesian or matrix‑based methods) would be needed for production‑level precision.
- Fit stability depends on binning choice; potential extension is to optimise the number of bins by minimising a loss that trades off statistical uncertainty against resolution.
