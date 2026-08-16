## Project Summary
This project forecasts rooftop photovoltaic (PV) power generation using sequence models and physics-motivated feature engineering on a three-year, multi station Hong Kong PV and weather dataset. It implements and compares three approaches:
- A baseline single‑station LSTM for 24‑hour horizon forecasting.
- A single-station GA-VMD-LSTM ensemble for 24-hour forecasting. This used Variational Mode Decomposition (VMD) --- optimised using a Genetic Algorithm (GA) --- to decompose power signals into intrinsic mode functions (IMFs), such that the sum of IMFs and a residual term summed to the total PV power of the timestep. Thus, an LSTM was trained to forecast each IMF, giving a GA-VMD-LSTM ensemble.
- A generalised multi-station LSTM that uses station metadata to forecast the next hour across 57 different PV stations.

On the main single‑station task, the GA‑VMD‑LSTM achieves a
$53\%$ RMSE reduction ($2383 \to 1122$), $51\%$ MAE reduction ($1258 \to 618$), and a $R^2$ improvement from $0.81$ to $0.96$ over the baseline LSTM over hourly 24-hour forecasts. The generalised model attains an average of $R^2 = 0.96$ for next-hour forecasting across 57 stations, with per-station $R^2$ as high as $0.99$.

## Environment
- Python 3.11
- Core libraries: 
(found here: https://datadryad.org/dataset/doi:10.5061/dryad.m37pvmd99)

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

# Single-station horizon forecasting (SQ19, 48h lookback / 24h horizon)
- Baseline LSTM: aggregate RMSE $2383$, MAE $1258$, $R^2 = 0.81$. <br> 
Performance drops sharply beyond short horizons and plateaus near mean prediction.
- GA-VMD-LSTM Ensemble: aggregate RMSE $1122$, MAE $618$, $R^2 = 0.96$. <br>
Reduces unexplained variance fraction ($1-R^2$) from $19.2\%$ to $4.3\%$ ($78\%$ reduction).
- Short horizons: GA-VMD-LSTM maintains $R^2>0.98$ through $t+8$, while baseline LSTM falls below 0.8 by that point.
- Per-IMF models: low‑frequency modes are highly predictable ($R^2>0.96$); high‑frequency modes show lower $R^2$, reflecting inherent randomness in rapid weather transients.

# Multi-station next-hour forecasting
- Generalised LSTM overall metrics: $R^2 = 0.9603$, capacity‑scaled RMSE $= 0.0368$ across 57 stations.
- Per‑station $R^2$ ranges from $0.81$ (data‑scarce stations) to $0.99$ (homogeneous SQ cluster), demonstrating strong generalisation conditioned on station metadata and capacity scaling.

# Lookback-horizon trade-off
- Increasing lookback beyond a full diurnal cycle offers limited benefit for horizons $h\leq 12$ hours, and can degrade performance for $h=24$ as long histories dilute sensitivity to recent weather.
- For very long horizons (e.g. 48h), both models plateau in $R^2$, indicating that forecast accuracy becomes limited by the inherent predictability of meteorological inputs rather than model capacity.

# Limitations and extensions
- Baseline and generalised LSTMs struggle to fully capture multi‑scale PV dynamics without VMD, showing weak validation‑loss convergence.
- GA‑VMD‑LSTM requires per‑station decomposition and multiple sub‑models, which increases computational cost and complexity of deployment.
- Natural extension: perform station‑specific VMD with GA‑optimised ($K,\alpha$) then train a shared multi‑station ensemble on the first $N$ low‑frequency IMFs while aggregating remaining high‑frequency modes into a residual channel.










