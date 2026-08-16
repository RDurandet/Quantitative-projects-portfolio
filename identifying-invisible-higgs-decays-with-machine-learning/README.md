# Identifying Invisible Higgs Decays with Machine Learning
This project investigates whether event-level machine learning can improve sensitivity to invisible Higgs-boson decays in the \(t\bar{t}H\) productionchannel at the Large Hadron Collider.

## Project overview
- Trains a Particle Transformer on simulated collider events to distinguish \(t\bar{t}H(H\rightarrow\mathrm{inv})\) signal from major Standard Model backgrounds (1:2000 signal-to-background ratio).
- Uses jet-level kinematic features, missing transverse momentum, and data-taking-period information as model inputs.
- Compares binary and multiclass classification strategies.
- Applies distance-correlation regularisation to control dependence between classifier scores and missing transverse momentum.
- Extracts projected branching-fraction sensitivity using profile-likelihood significance calculations.
- Validates the learned representation through feature-level analyses of physically expected angular correlations.

## Key results
- Multiclassifier AUC for the \(t\bar{t}H\) signal: **0.795**.
- Binary classifier AUC: **0.815**.
- Multiclassifier projected 95% CL upper limit on \(\mathrm{BR}(H\rightarrow\mathrm{inv})\): **31.3%** at \(138\,\mathrm{fb}^{-1}\), assuming a 10% flat background-yield uncertainty.
- Projected upper limit at \(3\,\mathrm{ab}^{-1}\): **17.5%** under the same uncertainty assumption.
- The model recovered physically expected relationships between classifier scores and angular separations involving missing transverse momentum, leading jets, and leading \(b\)-jets.
- The multiclassifier achieved a background distance correlation of 0.029, while the signal correlation was 0.496, consistent with the physical expectation that invisible Higgs decays produce genuine missing transverse momentum.

## Technical methods
- Particle Transformer with self-attention over up to 10 jets per event.
- Binary and multiclass classification.
- Cross-entropy loss with scheduled distance-correlation regularisation.
- AdamW optimisation, early stopping, learning-rate scheduling, and weighted training samples.
- ROC/AUC analysis, confusion matrices, score distributions, feature-level validation, and profile-likelihood significance calculations.

## Repository contents
- `report.pdf`: full technical report describing the analysis, methodology, validation, results, and limitations.

## Code availability
The implementation is not included in this repository because it was developed within a project environment using provided datasets and project infrastructure.
The report documents the model architecture, preprocessing, training procedure, statistical methodology, and validation sufficiently to explain the analysis.

## Scope and limitations
This is an expected-only analysis based on simulated events. It is not a result using observed collision data. A complete experimental analysis would require simulation-to-data validation, detector and modelling systematic uncertainties, and a full shape-based likelihood treatment.
