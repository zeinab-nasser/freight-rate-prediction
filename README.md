# Freight Rate Prediction

A machine learning project for predicting freight rates from shipment, route, geographic, temporal, and market-related features.

The project follows a complete machine learning workflow:

**Data Exploration → Feature Engineering → Preprocessing → Model Evaluation → Final Model Training → Prediction Generation**

---

## 📌 Project Overview

Freight pricing is influenced by multiple factors, including:

* Pickup and delivery locations
* Geographic distance
* Shipment distance
* Equipment type
* Shipment weight
* Market conditions
* Quote signals
* Temporal patterns

This project develops a regression-based machine learning solution to estimate freight rates (`posted_rate`) from available shipment and market information.

The final pipeline trains a **Linear Regression** model on the complete development dataset and generates:

1. Predictions for **12,000 validation loads**
2. Daily freight-rate predictions for **December 2025**
3. Reusable preprocessing and model artifacts

---

## 🎯 Objective

The primary objective is to build a reproducible machine learning pipeline capable of predicting freight rates while maintaining a clear separation between:

* Data exploration
* Feature engineering
* Preprocessing
* Model evaluation
* Final model training
* Prediction generation

The target variable is:

```text
posted_rate
```

---

# 📂 Project Structure

```text
freight-rate-prediction/
│
├── data/
│   ├── train-test.csv
│   ├── validation.csv
│   ├── validation-predictions-template.csv
│   └── december-chart-inputs.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering_preprocessing.ipynb
│   ├── 03_modeling_evaluation.ipynb
│   └── 04_final_model.ipynb
│
├── artifacts/
│   ├── preprocessor.joblib
│   ├── final_preprocessor.joblib
│   ├── final_linear_regression.joblib
│   ├── X_train_processed.npz
│   ├── X_val_processed.npz
│   ├── y_train.csv
│   └── y_val.csv
│
├── validation_predictions.csv
└── README.md
```

---

# 🔬 Machine Learning Workflow

## 1. Data Exploration

Notebook:

[`01_data_exploration.ipynb`](https://github.com/zeinab-nasser/freight-rate-prediction/blob/master/notebooks/01_data_exploration.ipynb)

The first notebook focuses on understanding the raw freight dataset, including:

* Dataset dimensions
* Feature types
* Missing values
* Numerical variables
* Categorical variables
* Target distribution
* Potential data-quality issues
* Relationships between important variables

This step establishes the foundation for subsequent preprocessing and modeling decisions.

---

# 🛠️ 2. Feature Engineering & Preprocessing

Notebook:

[`02_feature_engineering_preprocessing.ipynb`](https://github.com/zeinab-nasser/freight-rate-prediction/blob/master/notebooks/02_feature_engineering_preprocessing.ipynb)

Several domain-informed features were created to provide the models with additional information about routes, geography, time, and shipment characteristics.

### Date Features

The shipment date was transformed into:

* `month`
* `day_of_week`
* `day_of_month`
* `week_of_year`

These features allow the model to capture temporal patterns in freight pricing.

### Geographic Features

The project creates:

* `lat_diff`
* `lon_diff`
* `geographic_distance`

The geographic distance is calculated using the **Haversine formula**, providing an approximate straight-line distance between pickup and delivery coordinates.

### Route Efficiency

A logarithmic route-efficiency feature was created:

```text
log_route_efficiency =
log1p(distance / geographic_distance)
```

This captures the relationship between the reported shipment distance and geographic distance.

### Weight Features

Shipment weight was cleaned using its absolute value:

```text
weight_clean = abs(weight)
```

A normalized weight feature was also created:

```text
weight_per_mile = weight_clean / distance
```

This provides the model with information about shipment weight relative to route length.

---

# ⚙️ Preprocessing Pipeline

The preprocessing pipeline is implemented using a Scikit-learn `ColumnTransformer`.

Categorical features include:

```text
pickup
delivery
equipment
```

Numerical features include geographic, temporal, distance, market, quote, and weight-related variables.

The preprocessing pipeline produces:

```text
148 processed features
```

The same preprocessing logic is reused when transforming validation and December prediction data.

The fitted preprocessing pipeline is saved as:

```text
artifacts/final_preprocessor.joblib
```

---

# 📊 Dataset Split

The development dataset contains:

```text
48,000 rows
14 original columns
```

After feature engineering:

```text
48,000 rows
23 input features
```

After preprocessing:

```text
48,000 rows
148 processed features
```

For model evaluation, the development data was split into:

```text
Training set:   43,147 rows
Validation set:  4,853 rows
```

The separate assessment validation dataset contains:

```text
12,000 rows
```

---

# 🤖 3. Modeling & Evaluation

Notebook:

[`03_modeling_evaluation.ipynb`](https://github.com/zeinab-nasser/freight-rate-prediction/blob/master/notebooks/03_modeling_evaluation.ipynb)

Three modeling approaches were evaluated:

1. Linear Regression
2. Ridge Regression
3. XGBoost Regression

Two ensemble approaches were also tested.

---

## 📈 Evaluation Metrics

The models were evaluated using:

### MAE — Mean Absolute Error

Measures the average absolute difference between predicted and actual freight rates.

Lower values indicate smaller average prediction errors.

### RMSE — Root Mean Squared Error

Penalizes larger prediction errors more strongly than MAE.

Lower values indicate better performance.

### R² — Coefficient of Determination

Measures the proportion of target variance explained by the model.

Higher values indicate greater explanatory performance.

---

# 🧪 Model Results

| Model                                        |        MAE |       RMSE |         R² |
| -------------------------------------------- | ---------: | ---------: | ---------: |
| Linear Regression                            | **143.24** | **651.51** | **0.8183** |
| Ridge (α=1.0)                                |     147.57 |     652.03 |     0.8180 |
| XGBoost Baseline                             |     211.29 |     704.77 |     0.7874 |
| Ensemble (50% Linear + 50% XGBoost)          |     163.02 |     663.30 |     0.8117 |
| Weighted Ensemble (70% Linear + 30% XGBoost) |     150.61 |     654.84 |     0.8165 |

Based on the validation results, **Linear Regression was selected for the final training stage**, as it achieved the strongest validation metrics among the evaluated configurations.

---

# 🏆 Final Model

Notebook:

[`04_final_model.ipynb`](https://github.com/zeinab-nasser/freight-rate-prediction/blob/master/notebooks/04_final_model.ipynb)

The final model is:

```text
Linear Regression
```

The model is retrained using the complete development dataset:

```text
48,000 training samples
```

with:

```text
148 processed features
```

The final trained model is saved as:

```text
artifacts/final_linear_regression.joblib
```

The corresponding preprocessing pipeline is saved as:

```text
artifacts/final_preprocessor.joblib
```

---

# 🔮 Validation Predictions

The final model generates predictions for the separate validation dataset containing:

```text
12,000 loads
```

The output file is:

```text
validation_predictions.csv
```

with the following schema:

| Column           | Description                |
| ---------------- | -------------------------- |
| `load_id`        | Unique shipment identifier |
| `predicted_rate` | Predicted freight rate     |

The generated file was validated to ensure:

* 12,000 rows
* Correct column names
* No missing values
* No duplicate load IDs
* No non-positive predictions

---

# 📅 December 2025 Forecast

The project also generates daily freight-rate predictions for a representative route during December 2025.

Route:

```text
Lexington → Fort Wayne
```

Shipment characteristics:

```text
Distance: 360 miles
Equipment: Dry Van
Weight: 32,000
```

The December input contains:

```text
31 daily observations
```

The final model generates one predicted freight rate for each day.

Predictions range from approximately:

```text
$750.58 – $832.84
```

with an average predicted rate of approximately:

```text
$808.91
```

The predictions are stored in:

```text
data/december-chart-inputs.csv
```

---

# 🌎 Geographic Modeling

The project uses latitude and longitude information for both pickup and delivery locations.

The Haversine formula is used to estimate geographic distance:

```text
geographic_distance
```

This is combined with the provided shipment distance to create:

```text
log_route_efficiency
```

This allows the model to distinguish between geographic separation and the actual reported route distance.

---

# 📦 Saved Artifacts

The project saves reusable machine learning artifacts under:

```text
artifacts/
```

### Preprocessing

```text
preprocessor.joblib
final_preprocessor.joblib
```

### Final Model

```text
final_linear_regression.joblib
```

### Processed Training Data

```text
X_train_processed.npz
X_val_processed.npz
y_train.csv
y_val.csv
```

These artifacts make it possible to reuse the trained preprocessing and model without rebuilding the entire pipeline from scratch.

---

# 🔁 Reproducibility

The project is organized into sequential notebooks:

```text
01 → Data Exploration

02 → Feature Engineering & Preprocessing

03 → Modeling & Evaluation

04 → Final Model Training & Prediction
```

This structure makes the workflow easier to inspect, reproduce, and extend.

---

# ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/zeinab-nasser/freight-rate-prediction.git
```

Navigate to the project:

```bash
cd freight-rate-prediction
```

Install the required Python packages:

```bash
pip install numpy pandas scipy scikit-learn xgboost joblib jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

Run the notebooks sequentially:

```text
01_data_exploration.ipynb
02_feature_engineering_preprocessing.ipynb
03_modeling_evaluation.ipynb
04_final_model.ipynb
```

---

# 💡 Key Technical Decisions

### Why Linear Regression?

Multiple models were evaluated on the validation split.

Linear Regression produced the strongest validation performance among the tested models, while the XGBoost baseline did not outperform the linear baseline on this dataset.

### Why Feature Engineering?

Freight pricing depends on route characteristics, shipment properties, geographic relationships, market information, and time.

Feature engineering provides the model with explicit representations of these relationships rather than relying exclusively on the raw variables.

### Why a Separate Final Training Stage?

After model comparison, the selected model was retrained using the complete development dataset.

This allows the final model to use all available labeled development observations before generating predictions for the separate validation dataset.

---

# ⚠️ Limitations & Future Improvements

Potential future improvements include:

* Hyperparameter optimization for tree-based models
* Testing additional gradient boosting algorithms
* Cross-validation for more robust model comparison
* More advanced route-level features
* Historical route statistics
* Seasonal and holiday indicators
* Better treatment of extreme target values
* Prediction intervals / uncertainty estimation
* Model monitoring after deployment
* Automated inference pipeline
* API deployment for real-time predictions

---

# 📌 Final Results

The final project successfully produces:

```text
✓ Complete exploratory analysis
✓ Domain-informed feature engineering
✓ Reproducible preprocessing pipeline
✓ Model comparison
✓ Validation-based model selection
✓ Final Linear Regression model
✓ 12,000 validation predictions
✓ December 2025 daily predictions
✓ Saved preprocessing artifacts
✓ Saved trained model
✓ Submission-ready prediction file
```

---

# 👩‍💻 Author

**Zeinab Nasser**

Machine Learning / AI Engineer

GitHub:

https://github.com/zeinab-nasser

---

## 📚 Notebooks

| Notebook                                                                                                                                                                  | Purpose                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| [`01_data_exploration.ipynb`](https://github.com/zeinab-nasser/freight-rate-prediction/blob/master/notebooks/01_data_exploration.ipynb)                                   | Data exploration and understanding             |
| [`02_feature_engineering_preprocessing.ipynb`](https://github.com/zeinab-nasser/freight-rate-prediction/blob/master/notebooks/02_feature_engineering_preprocessing.ipynb) | Feature engineering and preprocessing          |
| [`03_modeling_evaluation.ipynb`](https://github.com/zeinab-nasser/freight-rate-prediction/blob/master/notebooks/03_modeling_evaluation.ipynb)                             | Model training and evaluation                  |
| [`04_final_model.ipynb`](https://github.com/zeinab-nasser/freight-rate-prediction/blob/master/notebooks/04_final_model.ipynb)                                             | Final model training and prediction generation |
