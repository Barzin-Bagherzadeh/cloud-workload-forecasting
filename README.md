# Cloud Workload Forecasting & Analysis

## 1. Project Overview

This project focuses on forecasting cloud workload demand using CPU utilization data from the Microsoft Azure Public Dataset V2.

The objective is to analyze historical workload patterns, develop forecasting models, evaluate their performance against a baseline, and provide insights that can support cloud infrastructure capacity planning and resource allocation.

The project forecasts two workload metrics:

- `active_vm_count`: Number of active virtual machines at each timestamp.
- `aggregate_cpu_utilization`: Mean CPU utilization across active virtual machines at each timestamp.

## 2. Dataset

This project uses a curated subset of the Microsoft Azure Public Dataset V2 provided for the Cloud Workload Forecasting assignment.

### 2.1. Dataset Contents

The supplied dataset contains:

- 10,000 virtual machines.
- 1,340,000 CPU measurements.
- 134 observations per virtual machine.
- A sampling interval of 5 minutes.
- Approximately 11.1 hours of continuous observations per virtual machine.

The dataset archive contains the following files:

- `vm_cpu_readings.csv.gz`: VM-level CPU utilization readings.
- `vm_metadata.csv.gz`: Additional virtual machine metadata.
- `DATA_CARD.md`: Dataset documentation.
- `SHA256SUMS`: Checksums for verifying the dataset files.

### 2.2. Download and Setup

The dataset is distributed separately and is not included in this repository.

Download the dataset from the official GitHub Release:

**[Azure Workload Dataset v1 – GitHub Release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)**

To prepare the dataset:

1. Open the GitHub Release page linked above.
2. Download the `data-v1.tar.gz` archive from the release assets.
3. Extract the archive.
4. Place the extracted `challenge_data` directory in the root directory of this project.
5. Verify the downloaded dataset files using the SHA256 checksums provided in the archive.

After extraction, the project should have the following structure:

```text
cloud-workload-forecasting/
├── challenge_data/
│   ├── vm_cpu_readings.csv.gz
│   ├── vm_metadata.csv.gz
│   ├── DATA_CARD.md
│   └── SHA256SUMS
├── notebooks/
├── src/
├── tests/
├── pyproject.toml
└── README.md
```

The `challenge_data/` directory is excluded from Git because the dataset is distributed separately and should be downloaded using the official release link.

## 3. Installation

Python 3.11 or later is required.

### 3.1. Create a Virtual Environment

From the project root, create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
```

**Linux and macOS:**

```bash
source .venv/bin/activate
```

### 3.2. Install Dependencies

Install the project and its dependencies:

```bash
pip install -e .
```

To install the development dependencies, including pytest, use:

```bash
pip install -e ".[dev]"
```

## 4. Data Preparation

The CPU readings are grouped by timestamp to construct two target time series.

The targets are calculated as follows:

- `active_vm_count`: Number of unique virtual machines (`vm_id`) with valid CPU observations at each timestamp.
- `aggregate_cpu_utilization`: Mean of the `avg_cpu` values of the virtual machines with valid CPU observations at each timestamp.

Rows with missing timestamps, VM identifiers, or CPU utilization values are excluded before aggregation.

The resulting time series is sorted chronologically.

### 4.1. Training and Holdout Split

The final 24 observations are reserved as the holdout test set. All preceding observations are used for model development and training.

For the supplied dataset, this corresponds to:

- Total observations: 134.
- Training and development observations: 110.
- Holdout observations: 24 (2 hours).

The holdout period is kept separate from model development and is used only for final evaluation. Its observed target values are not used for model fitting or forecasting.

## 5. Forecasting Approach

### 5.1. Baseline Model

A persistence baseline is used as a reference for evaluating CPU utilization forecasts.

The persistence model predicts that future CPU utilization will remain equal to the last observed training value.

This baseline is evaluated using the same time-aware validation folds as the main forecasting model.

### 5.2. Main Forecasting Model

The main forecasting model uses `SGDRegressor` from scikit-learn.

Two input features are used:

- `timestamp`
- `median_4`: Median of the previous four CPU utilization observations.

Before training, the input features are standardized using `StandardScaler`.

The model is trained using the training portion of the time series.

For multi-step forecasting, predictions are generated recursively. After each prediction, the predicted value is added to the available history and used to calculate the median feature for the next forecasting step.

This process is repeated until all 24 holdout timestamps have been forecast.

### 5.3. Active VM Count Forecast

The active VM count forecast uses a simple adaptive approach based on the training data:

- If the active VM count remains constant throughout the training period, that constant value is used for all future timestamps.
- If the active VM count varies, the last observed training value is used as a persistence forecast for all future timestamps.

For the supplied dataset, the active VM count remains constant at 10,000. Therefore, the forecast is 10,000 for all 24 holdout timestamps.

This approach does not use observed values from the holdout period.

## 6. Model Validation and Evaluation

A time-aware expanding-window validation strategy is used to evaluate the CPU forecasting model.

The validation configuration is:

- Minimum training size: 60 observations.
- Validation horizon: 20 observations.
- Step size: 10 observations.

Using the first 110 observations, this configuration produces four chronological validation folds.

The following evaluation metrics are used:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted values.
- **RMSE (Root Mean Squared Error):** Penalizes larger forecasting errors more heavily.
- **SMAPE (Symmetric Mean Absolute Percentage Error):** Measures forecasting error as a percentage using a symmetric denominator.

Normalized MAE is also used to compare the main model with the persistence baseline. It is calculated as the main model's MAE divided by the baseline's MAE.

A normalized MAE below 1 indicates that the main model performs better than the baseline.

### 6.1. Cross-Validation Results

The following table shows the average results across the four validation folds:

| Metric | SGDRegressor | Persistence |
|---|---:|---:|
| MAE | 0.144554 | 0.201091 |
| RMSE | 0.172985 | 0.233117 |
| SMAPE | 1.532839% | 2.121962% |

During cross-validation, SGDRegressor achieves lower average errors than the persistence baseline across all three metrics.

### 6.2. Final Holdout Results

After model development, the final 24 observations are evaluated separately.

| Metric | SGDRegressor | Persistence |
|---|---:|---:|
| MAE | 0.226073 | 0.146782 |
| RMSE | 0.270413 | 0.184785 |
| SMAPE | 2.455808% | 1.585054% |

Although SGDRegressor performs better during cross-validation, the persistence baseline achieves lower errors on the final holdout period across all three metrics.

This indicates that the main model's validation performance does not fully generalize to the unseen holdout period. It also demonstrates the importance of evaluating forecasting models on a separate chronological test set.

## 7. Prediction Output

The forecasting pipeline generates a CSV file named `predictions.csv`.

The output contains exactly 24 rows and the following columns, in the required order:

| Column | Description |
|---|---|
| `timestamp` | Timestamp of the forecast |
| `active_vm_count` | Predicted number of active virtual machines |
| `aggregate_cpu_utilization` | Predicted aggregate CPU utilization |

The output is saved to:

```text
predictions/predictions.csv
```

Before saving the file, the pipeline validates the predictions to check that:

- Exactly 24 rows are present.
- The required columns are present in the correct order.
- No missing or non-finite values exist.
- Timestamps are in chronological order.
- No duplicate timestamps exist.
- Active VM counts are non-negative.
- CPU utilization values are between 0 and 100.

The CSV file is saved without an additional index column.

## 8. Running the Project

After downloading and extracting the dataset into the `challenge_data/` directory, run the following command from the project root:

```bash
python -m cwfc.predict --data challenge_data --output predictions/predictions.csv
```

The pipeline:

1. Builds the target time series from the CPU readings.
2. Separates the final 24 observations from the training data.
3. Trains the CPU forecasting model using the training data.
4. Generates recursive CPU utilization forecasts.
5. Forecasts the active VM count using the adaptive approach.
6. Validates the prediction output.
7. Saves the resulting CSV file.

The generated prediction file does not require manual editing.

## 9. Running Tests

To run the test suite, execute:

```bash
pytest -v
```

The tests cover:

- Evaluation metric calculations.
- Time-aware validation splits.
- Prediction output validation, including row count, duplicate timestamps, CPU range, and non-negative active VM counts.

## 10. Operational Considerations

Forecasting workload demand can support infrastructure teams in making decisions about resource allocation, capacity planning, and scaling.

A practical operational strategy could include:

- Monitoring aggregate CPU utilization and active VM count continuously.
- Scaling up when CPU utilization remains elevated over a sustained period.
- Maintaining sufficient spare capacity to handle unexpected workload increases.
- Scaling down gradually when demand remains low.
- Evaluating forecasting errors during periods of high workload to reduce the risk of under-provisioning.

The final holdout results show that the persistence baseline outperforms the main model on the evaluated CPU utilization series. Therefore, SGDRegressor should not be assumed to provide a more reliable operational forecast without further investigation and improvement.

The active VM count forecast also relies on a simple assumption: constant values are carried forward when the training series is constant; otherwise, the last observed training value is used. More sophisticated forecasting methods may be needed when the number of active virtual machines changes substantially over time.