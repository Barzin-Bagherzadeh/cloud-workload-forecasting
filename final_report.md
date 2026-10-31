# Cloud Workload Forecasting & Analysis

## 1. Introduction

Cloud infrastructure providers must continuously monitor workload demand to make effective decisions about resource allocation, capacity planning, and scaling. Accurate workload forecasting can help infrastructure teams anticipate demand changes, maintain sufficient computational resources, and avoid unnecessary infrastructure costs.

The objective of this project is to analyze historical CPU utilization data from the Microsoft Azure Public Dataset V2 and develop a forecasting pipeline for short-term cloud workload demand.

Two workload metrics are considered:

- **Active VM Count:** The number of active virtual machines observed at each timestamp.
- **Aggregate CPU Utilization:** The mean CPU utilization across active virtual machines at each timestamp.

The project follows an end-to-end forecasting workflow, including data preparation, exploratory analysis, baseline development, model training, time-aware validation, final holdout evaluation, and operational recommendations.

The main forecasting model uses `SGDRegressor` from scikit-learn. Its performance is compared against a persistence baseline to determine whether the more complex model provides a meaningful improvement over a simple forecasting approach.

## 2. Dataset and Preprocessing

### 2.1. Dataset Description

The project uses a curated subset of the Microsoft Azure Public Dataset V2 provided for this assignment.

The supplied dataset contains:

| Property | Value |
|---|---:|
| Number of virtual machines | 10,000 |
| Total CPU measurements | 1,340,000 |
| Observations per VM | 134 |
| Sampling interval | 5 minutes |
| Approximate observation duration | 11.1 hours |

The CPU readings are stored in `vm_cpu_readings.csv.gz`, while additional VM information is provided in `vm_metadata.csv.gz`.

Each CPU observation contains information about a virtual machine's CPU utilization at a particular timestamp.

### 2.2. Target Construction

The original VM-level observations are grouped by timestamp to construct two aggregate time series.

**Active VM Count**

The active VM count is calculated by counting the number of unique virtual machines observed at each timestamp:

\[
active\_vm\_count_t =
\left|\{vm\_id : vm\_id \text{ is observed at } t\}\right|
\]

**Aggregate CPU Utilization**

Aggregate CPU utilization is calculated as the arithmetic mean of the VM-level `avg_cpu` measurements at each timestamp:

\[
aggregate\_cpu\_utilization_t =
\frac{1}{N_t}\sum_{i=1}^{N_t} avg\_cpu_{i,t}
\]

where \(N_t\) represents the number of virtual machines with valid observations at timestamp \(t\).

The resulting time series is sorted chronologically before being used for forecasting.

### 2.3. Training and Holdout Separation

The final 24 observations are reserved as the holdout test set. All preceding observations are used for model development and training.

For the supplied dataset, the split is:

| Dataset portion | Number of observations |
|---|---:|
| Training and development | 110 |
| Final holdout | 24 |
| Total | 134 |

Since each observation represents a five-minute interval, the final 24 observations correspond to a two-hour forecasting horizon.

The holdout observations are excluded from model fitting and model development. They are used only for the final evaluation.

## 3. Forecasting Methods

### 3.1. Persistence Baseline

A persistence forecasting model is used as the baseline.

The persistence approach assumes that the future value of a time series will remain equal to its most recently observed training value.

For CPU utilization, the forecast is:

\[
\hat{y}_{t+h}=y_t
\]

where \(y_t\) is the last observed training value and \(h\) represents a future forecasting step.

Although this method is simple, it provides an important reference point. A more complex forecasting model should demonstrate that it can improve upon this basic assumption.

The persistence baseline is evaluated using the same time-aware validation strategy as the main model.

### 3.2. Main Model: SGDRegressor

The main forecasting model uses `SGDRegressor`, a linear regression estimator trained using stochastic gradient descent.

Two input features are used:

- `timestamp`: The timestamp associated with the observation.
- `median_4`: The median of the previous four CPU utilization observations.

The median feature provides information about recent workload behavior and is less sensitive to individual extreme observations than a simple average.

Before model training, the input features are standardized using `StandardScaler`.

The model learns a relationship between the input features and aggregate CPU utilization.

### 3.3. Recursive Multi-step Forecasting

The forecasting pipeline generates predictions recursively for the 24 future timestamps.

For each forecasting step:

1. The median of the most recent four available CPU values is calculated.
2. The timestamp and median feature are passed to the trained model.
3. The model generates the next CPU utilization prediction.
4. The predicted value is appended to the available history.
5. The updated history is used to calculate the median feature for the next step.

This process continues until predictions have been generated for all 24 holdout timestamps.

The recursive approach allows the model to produce multi-step forecasts without using the actual CPU utilization values from the holdout period.

Predicted CPU utilization values are constrained to the valid range of 0 to 100.

### 3.4. Active VM Count Forecasting

The active VM count forecast uses a simple adaptive approach based on the training data.

If the active VM count remains constant throughout the training period, that constant value is used for all future timestamps.

If the active VM count varies, the last observed training value is used as a persistence forecast.

For the supplied dataset, the active VM count remains constant at 10,000 during the training period. Therefore, the forecast is 10,000 for all 24 holdout timestamps.

This approach is intentionally simple and does not use observed holdout values.

## 4. Validation Strategy and Evaluation Metrics

### 4.1. Time-aware Cross-validation

A time-aware expanding-window validation strategy is used to evaluate the CPU forecasting model.

Unlike random train-test splitting, this approach preserves the chronological order of observations. This is important for time series forecasting because future observations must not be used to predict earlier ones.

The validation configuration is:

| Parameter | Value |
|---|---:|
| Minimum training size | 60 observations |
| Validation horizon | 20 observations |
| Step size | 10 observations |
| Number of validation folds | 4 |

At each fold, the training window expands while the validation period moves forward in time.

This strategy evaluates the model across multiple chronological validation periods using only the first 110 observations.

### 4.2. Evaluation Metrics

Three primary metrics are used to evaluate forecasting performance.

**Mean Absolute Error (MAE)**

MAE measures the average absolute difference between actual and predicted values.

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

Lower MAE values indicate more accurate predictions on average.

**Root Mean Squared Error (RMSE)**

RMSE measures the square root of the average squared prediction errors.

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

Because errors are squared before averaging, RMSE gives greater weight to larger prediction errors.

**Symmetric Mean Absolute Percentage Error (SMAPE)**

SMAPE expresses forecasting errors as a percentage using a symmetric denominator.

\[
SMAPE=\frac{100}{n}\sum_{i=1}^{n}
\frac{2|y_i-\hat{y}_i|}
{|y_i|+|\hat{y}_i|}
\]

A small numerical constant is used in the implementation to avoid division by zero.

Lower values of all three metrics indicate better forecasting performance.

Normalized MAE is also used to compare the main model with the persistence baseline:

\[
Normalized\ MAE =
\frac{MAE_{\text{model}}}{MAE_{\text{baseline}}}
\]

A value below 1 indicates that the main model performs better than the baseline, while a value above 1 indicates that the baseline performs better.

## 5. Results and Discussion

### 5.1. Cross-validation Results

The following table presents the average results across the four chronological validation folds.

| Metric | SGDRegressor | Persistence |
|---|---:|---:|
| MAE | 0.144554 | 0.201091 |
| RMSE | 0.172985 | 0.233117 |
| SMAPE | 1.532839% | 2.121962% |

During cross-validation, SGDRegressor achieves lower average errors than the persistence baseline across all three metrics.

The results suggest that the recent CPU history and timestamp features provide useful information for predicting CPU utilization during the validation periods.

However, cross-validation performance alone is not sufficient to establish that a model will perform well on unseen future observations.

### 5.2. Final Holdout Results

After the model development process, both forecasting approaches are evaluated on the final 24 observations.

| Metric | SGDRegressor | Persistence |
|---|---:|---:|
| MAE | 0.226073 | 0.146782 |
| RMSE | 0.270413 | 0.184785 |
| SMAPE | 2.455808% | 1.585054% |

The persistence baseline achieves lower errors than SGDRegressor across all three evaluation metrics.

This is an important result because it shows that the main model's performance during cross-validation does not fully generalize to the final holdout period.

Although SGDRegressor performs better on average during the validation folds, its forecasting errors increase on the unseen holdout period. The persistence baseline, in contrast, maintains better performance on the final evaluation data.

### 5.3. Interpretation of Results

The difference between cross-validation and holdout results highlights an important challenge in time series forecasting: workload patterns may change over time, and a model that performs well during earlier periods may not perform equally well during a later period.

The main model relies on a linear relationship between timestamp, recent median CPU utilization, and the target variable. This assumption may not capture every change in workload behavior.

The persistence baseline benefits from the relatively stable CPU utilization observed near the end of the training period.

Therefore, the results do not support the assumption that a more complex model necessarily provides more accurate forecasts.

For this dataset and forecasting horizon, persistence is the better-performing approach on the final holdout period.

## 6. Operational Recommendations

Forecasting workload demand can support infrastructure teams in making decisions about capacity planning, resource allocation, and scaling.

Based on the forecasting approach and evaluation results, the following operational considerations are recommended:

- **Continuous monitoring:** Monitor aggregate CPU utilization and active VM count to identify workload changes.
- **Scale-up decisions:** Consider scaling up when CPU utilization remains elevated over a sustained period rather than reacting to isolated spikes.
- **Capacity headroom:** Maintain sufficient spare capacity to accommodate unexpected workload increases and reduce the risk of under-provisioning.
- **Gradual scale-down:** Reduce allocated resources gradually when demand remains low to avoid unnecessary infrastructure costs.
- **Peak-period evaluation:** Pay particular attention to forecasting errors during periods of high CPU utilization, since underestimating demand can create operational risks.
- **Baseline comparison:** Continue comparing forecasting models against simple baselines to ensure that additional model complexity provides measurable benefits.

The final holdout results indicate that the persistence baseline is currently more reliable than SGDRegressor for the evaluated CPU utilization series.

Consequently, the persistence approach is the more suitable of the two evaluated methods for this particular holdout period. The main model would require further investigation and improvement before being preferred for operational deployment.

## 7. Limitations and Future Improvements

Several limitations should be considered when interpreting the results.

**Limited observation period**

The supplied dataset contains approximately 11.1 hours of observations per VM. This relatively short history limits the ability to identify longer-term workload patterns, daily cycles, and recurring seasonal behavior.

**Linear model assumptions**

SGDRegressor uses a linear relationship between the input features and the target. More complex workload behavior may not be adequately represented by this approach.

**Recursive forecasting errors**

In recursive forecasting, each prediction becomes part of the input history for subsequent predictions. Consequently, an early prediction error can influence later forecasting steps.

**Active VM count forecasting**

The active VM count approach relies on a constant-value or persistence assumption. Although this is appropriate for the supplied dataset, it may be insufficient when the number of active virtual machines changes substantially over time.

Potential future improvements include:

- Testing additional time-series forecasting models.
- Investigating alternative historical features and rolling statistics.
- Evaluating different forecasting horizons.
- Exploring workload patterns over longer observation periods.
- Developing a dedicated forecasting model for active VM count.
- Investigating model performance during high-utilization periods.

These improvements could help produce more reliable forecasts across different workload conditions.

## 8. Conclusion

This project developed an end-to-end cloud workload forecasting pipeline using a curated subset of the Microsoft Azure Public Dataset V2.

Two workload metrics were constructed: active VM count and aggregate CPU utilization. A persistence baseline and an SGDRegressor-based forecasting model were implemented and evaluated using chronological validation and a separate final holdout period.

The cross-validation results showed that SGDRegressor achieved lower average forecasting errors than the persistence baseline. However, the final holdout evaluation showed that persistence outperformed the main model across MAE, RMSE, and SMAPE.

This difference demonstrates the importance of evaluating forecasting models on unseen future observations and not relying exclusively on validation performance.

Overall, the project provides a reproducible forecasting pipeline, an evaluation framework, and practical considerations for cloud infrastructure capacity planning. The results also emphasize that simple forecasting methods can be highly competitive when workload behavior is relatively stable.