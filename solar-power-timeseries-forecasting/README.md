## Project Summary
This project forecasts rooftop photovoltaic (PV) power generation using sequence models and physics-motivated feature engineering on a three-year, multi station Hong Kong PV and weather dataset (found here: https://datadryad.org/dataset/doi:10.5061/dryad.m37pvmd99). It implements and compares three approaches:
- A baseline single‑station LSTM for 24‑hour horizon forecasting.
- A single-station GA-VMD-LSTM ensemble for 24-hour forecasting. This used Variational Mode Decomposition (VMD) --- optimised using a Genetic Algorithm (GA) --- to decompose power signals into intrinsic mode functions (IMFs), such that the sum of IMFs and a residual term summed to the total PV power of the timestep. Thus, an LSTM was trained to forecast each IMF, giving a GA-VMD-LSTM ensemble.
- A generalised multi-station LSTM that uses station metadata to forecast the next hour across 57 different PV stations.

On the main single‑station task, the GA‑VMD‑LSTM achieves a
$53\%$ RMSE reduction ($2383 \to 1122$), $51\%$ MAE reduction ($1258 \to 618$), and a $R^2$ improvement from $0.81$ to $0.96$ over the baseline LSTM over hourly 24-hour forecasts. The generalised model attains an average of $R^2 = 0.96$ for next-hour forecasting across 57 stations, with per-station $R^2$ as high as $0.99$.

## Motivation and problem
- **Goal**: train physically-grounded timeseries models that can forecast PV power from local weather data and PV power from a set lookback window.
- **Challenges addressed**: non‑linear multi‑timescale dynamics (diurnal, seasonal, and transient weather), station heterogeneity, data leakage in time series, and overfitting in high‑capacity recurrent networks.

## Data preprocessing
- Uses the high‑resolution rooftop PV + on‑site weather dataset covering 60 grid‑connected stations in Hong Kong (2021–2023).
- Resamples power and weather measurements to hourly resolution with mean aggregation to preserve physical interpretation of time‑averaged power and irradiance.
- Merges PV and weather data on timestamp; drops rows with NaNs to avoid training on partially missing sequences.
- Applies a chronological 75%–15%–15% train/validation/test split to prevent leakage of future information into training.

## Physics-motivated feature engineering
- **Lagged power**: uses sliding windows of historical power as the dominant predictor; captures short‑term continuity and provides context for meteorological features.
- **Weather features**: selects irradiance, temperature, and relative humidity on physical grounds (primary driver of PV output, panel efficiency via material properties, and condensation/cloud proxies), dropping second‑order metrics with limited incremental value.
- **Solar geometry**: adds solar zenith and azimuth via `pvlib`, cyclically encoded to respect angle periodicity (ensuring 359° and 0° are treated as neighbours).
- **Station metadata (multi‑station model)**: uses altitude, capacity, tilt, and azimuth to encode station physics (air mass, installed size, incidence angle), cyclically encoding azimuth and normalising power by capacity to learn a shared fractional‑utilisation scale.

## Sequence construction and normalisation
- Builds sliding‑window sequences with configurable lookback (past hours) and horizon (future steps); ensures windows are adjacent in time to avoid gaps.
- For single‑station models: MinMax scales all features to $[0,1]$, matching LSTM gate activations (sigmoid/tanh) and avoiding saturation.
- For multi‑station models: normalises power by station capacity while meteorological and static features retain MinMax scaling; enforces a unified output scale across stations.

## Models

# Baseline single-station LSTM
- 2‑layer stacked LSTM (hidden size 64) feeding a 2‑layer fully‑connected head with intermediate ReLU for direct multi‑step prediction (24‑hour horizon) without autoregressive roll‑out.
- Uses dropout of 0.2 between LSTM layers and 0.1 in the FC head to control overfitting while preserving learned sequence structure.
- Trains with MSE loss (quadratic penalty on large errors), Adam optimiser, ReduceLROnPlateau scheduler (LR  $0.001$ with floor $1\times 10^{-4}$) and early stopping on validation loss.

# GA-VMD-LSTM Ensemble
- Applies Variational Mode Decomposition to split the power signal into $K$ band‑limited intrinsic mode functions (IMFs) plus a residual, each centred on a distinct frequency band. VMD was chosen over EMD to avoid mode mixing.
- Optimises VMD hyperparameters ($K,\alpha$) via a genetic algorithm on the training set, using mean envelope entropy as the fitness function to favour clean, well‑separated modes; converges to $K=7, \alpha = 100$ for the main station (SQ19).
- Trains $K+1$ IMF‑specific LSTMs with capacity allocated by frequency: deeper, larger models for low‑frequency components (seasonality, diurnal cycles) and lighter models for high‑frequency weather transients; sums IMF predictions to reconstruct total power.

# Generalised multi-station LSTM
- Extends the baseline architecture to all stations by feeding static metadata through a small linear network with tanh activation, then using the resulting embedding to initialise $h_0$ and $c_0$ for all LSTM layers.
- This seeds the LSTM’s memory with station identity before time‑varying inputs arrive, preventing conflation of static and dynamic information.
- Uses AdamW for decoupled weight decay, higher dropout ($0.4$ between LSTM layers), adjusted LR scheduling, and longer early‑stopping patience to manage the more complex, heterogeneous multi‑station setting.

## Results

#Single-station horizon forecasting (SQ19, 48h lookback / 24h horizon)
- 










