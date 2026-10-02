# Freight Rate Prediction

A reproducible machine learning pipeline for predicting freight rates from shipment characteristics, route information, geographic relationships, temporal patterns, and market signals.

The project implements an end-to-end regression workflow covering:

**Data Exploration → Feature Engineering → Preprocessing → Model Evaluation → Final Training → Validation Prediction → December Forecast**

---

## 📌 Project Overview

Freight rates are influenced by a combination of route, shipment, geographic, temporal, and market-related factors.

This project develops a machine learning solution to estimate the target variable:

```text
posted_rate
```

The workflow is designed to be reproducible and modular, with dedicated notebooks for exploration, preprocessing, model evaluation, and final prediction generation.

The final pipeline produces:

* A trained Linear Regression model
* A reusable preprocessing pipeline
* Predictions for **12,000 validation loads**
* Daily freight-rate predictions for **December 2025**
* Reusable serialized model artifacts
* A submission-ready prediction file
* A December forecast visualization

---

# 🎯 Objective

The primary objective is to build a reliable regression pipeline capable of predicting freight rates while maintaining a clear separation between:

1. Data exploration
2. Feature engineering
3. Data preprocessing
4. Model evaluation
5. Model selection
6. Final model training
7. Prediction generation
8. Output validation

The target variable is:

```text
posted_rate
```

---

# 🏗️ Project Architecture

The project is organized into four main stages:

```text
Raw Data
   │
   ▼
Data Exploration
   │
   ▼
Feature Engineering
   │
   ▼
Preprocessing Pipeline
   │
   ▼
Model Evaluation
   │
   ▼
Model Selection
   │
   ▼
Final Training
   │
   ├──► Validation Predictions
   │
   └──► December 2025 Forecast
```

---

# 📂 Project Structure

```text
freight-rate-prediction/
│
├── .gitignore
│
├── artifacts/
│   ├── final_linear_regression.joblib
│   ├── final_preprocessor.joblib
│   ├── preprocessor.joblib
│   ├── X_train_processed.npz
│   ├── X_val_processed.npz
│   ├── y_train.csv
│   └── y_val.csv
│
├── data/
│   ├── december-chart-inputs.csv
│   ├── train-test.csv
│   ├── validation-predictions-template.csv
│   └── validation.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering_preprocessing.ipynb
│   ├── 03_modeling_evaluation.ipynb
│   └── 04_final_model.ipynb
│
├── scorer_results/
│   └── candidate_december.png
│
├── src/
│   ├── features.py
│   ├── model.py
│   ├── prediction.py
│   ├── preprocessing.py
│   └── __pycache__/
│
├── spotter-ml-env/
│
├── README.md
├── requirements.txt
├── score.py
└── validation_predictions.csv
```

> `spotter-ml-env/` is the local Python virtual environment and is not required for reproducing the project on another machine.

---

# 🔬 1. Data Exploration

Notebook:

```text
notebooks/01_data_exploration.ipynb
```

The exploration stage focuses on understanding the raw freight dataset before modeling.

The analysis includes:

* Dataset dimensions
* Column types
* Missing-value inspection
* Numerical feature analysis
* Categorical feature analysis
* Target distribution
* Data-quality checks
* Relationships between important variables
* Identification of potential modeling issues

The development dataset contains:

```text
48,000 rows
14 original columns
```

The original columns include shipment identifiers, pickup and delivery information, geographic coordinates, distance, equipment, weight, date, market signals, and the target freight rate.

---

# 🛠️ 2. Feature Engineering & Preprocessing

Notebook:

```text
notebooks/02_feature_engineering_preprocessing.ipynb
```

Feature engineering incorporates domain-informed representations of route geometry, time, shipment characteristics, and freight movement.

## Date Features

The shipment date is transformed into:

```text
month
day_of_week
day_of_month
week_of_year
```

These features allow the model to capture temporal patterns in freight pricing.

---

## Geographic Features

Pickup and delivery coordinates are used to derive:

```text
lat_diff
lon_diff
geographic_distance
```

`geographic_distance` is calculated using the Haversine formula to estimate the straight-line distance between pickup and delivery locations.

---

## Route Efficiency

A logarithmic route-efficiency feature is created from the relationship between reported shipment distance and geographic distance:

```text
log_route_efficiency =
log1p(distance / geographic_distance)
```

This provides the model with an additional representation of route geometry.

---

## Weight Features

Shipment weight is cleaned using its absolute value:

```text
weight_clean = abs(weight)
```

A normalized weight feature is then calculated:

```text
weight_per_mile =
weight_clean / distance
```

This represents shipment weight relative to route length.

---

# ⚙️ Preprocessing Pipeline

The preprocessing workflow uses Scikit-learn's `ColumnTransformer`.

Categorical variables include:

```text
pickup
delivery
equipment
```

Numerical variables include geographic, temporal, distance, market, quote, and weight-related features.

The engineered dataset contains:

```text
48,000 rows
23 input features
```

After preprocessing:

```text
48,000 rows
148 processed features
```

The fitted preprocessing pipeline is saved as:

```text
artifacts/preprocessor.joblib
```

The final fitted preprocessing pipeline is saved as:

```text
artifacts/final_preprocessor.joblib
```

The same preprocessing logic is reused for validation and December prediction data.

---

# 📊 Dataset Split

For model evaluation, the development dataset was split into:

```text
Training set:    43,147 rows
Validation set:   4,853 rows
```

Processed matrices:

```text
X_train_processed: (43,147, 148)
X_val_processed:  (4,853, 148)
```

The separate assessment validation dataset contains:

```text
12,000 rows
```

---

# 🤖 3. Modeling & Evaluation

Notebook:

```text
notebooks/03_modeling_evaluation.ipynb
```

Multiple regression approaches were evaluated using the same processed validation split.

The evaluated approaches were:

1. Linear Regression
2. Ridge Regression
3. XGBoost Regression
4. 50/50 Linear Regression + XGBoost Ensemble
5. 70/30 Weighted Linear Regression + XGBoost Ensemble

---

# 📏 Evaluation Metrics

## MAE — Mean Absolute Error

MAE measures the average absolute difference between predicted and actual freight rates.

Lower values indicate smaller average prediction errors.

---

## RMSE — Root Mean Squared Error

RMSE penalizes larger prediction errors more strongly than MAE.

Lower values indicate smaller squared prediction errors.

---

## R² — Coefficient of Determination

R² measures the proportion of target variance explained by the model.

Higher values indicate greater explanatory performance.

---

# 📈 Model Results

| Model                                        |        MAE |       RMSE |         R² |
| -------------------------------------------- | ---------: | ---------: | ---------: |
| **Linear Regression**                        | **140.61** |     650.55 |     0.8189 |
| Ridge (α = 1.0)                              |     140.90 | **650.53** | **0.8189** |
| XGBoost Baseline                             |     211.29 |     704.77 |     0.7874 |
| Ensemble (50% Linear + 50% XGBoost)          |     161.32 |     662.97 |     0.8119 |
| Weighted Ensemble (70% Linear + 30% XGBoost) |     148.57 |     654.29 |     0.8168 |

### Model Selection

Linear Regression achieved the lowest MAE among all evaluated configurations:

```text
MAE = 140.61
RMSE = 650.55
R² = 0.8189
```

Ridge Regression produced a very similar validation result and achieved a slightly lower RMSE, but its MAE was higher than Linear Regression.

The XGBoost baseline did not outperform the linear models on this validation split.

The ensemble experiments were retained as documented alternatives for further experimentation.

Based on the validation results and the project's primary focus on MAE, **Linear Regression was selected as the final model**.

---

# 🏆 4. Final Model

Notebook:

```text
notebooks/04_final_model.ipynb
```

The selected model is:

```text
Linear Regression
```

The final model is retrained using the complete development dataset:

```text
48,000 training samples
```

with:

```text
148 processed features
```

The final model contains:

```text
148 coefficients
```

and is saved as:

```text
artifacts/final_linear_regression.joblib
```

The corresponding fitted preprocessing pipeline is saved as:

```text
artifacts/final_preprocessor.joblib
```

---

# 🔮 Validation Predictions

The final model generates predictions for the separate assessment validation dataset.

Dataset size:

```text
12,000 loads
```

The submission file is:

```text
validation_predictions.csv
```

Output schema:

| Column           | Description                |
| ---------------- | -------------------------- |
| `load_id`        | Unique shipment identifier |
| `predicted_rate` | Predicted freight rate     |

The final output was validated to ensure:

```text
Rows:                    12,000
Columns:                 2
Missing predictions:     0
Duplicate load IDs:     0
Non-positive predictions: 0
```

The model initially produced 11 non-positive raw predictions. These were handled during the final output validation step by applying a lower bound of `1.0` to ensure valid positive freight-rate predictions.

---

# 📅 December 2025 Forecast

The final model was also used to generate daily freight-rate predictions for a representative route during December 2025.

### Route

```text
Lexington → Fort Wayne
```

### Shipment Characteristics

```text
Distance: 360 miles
Equipment: Dry Van
Weight: 32,000
```

The forecast contains:

```text
31 daily observations
```

Because December market observations were not present in the development dataset, the December market features were populated using the mean market information from the last available training month, October 2025.

The resulting December predictions were generated using the same fitted preprocessing pipeline and final Linear Regression model.

### Forecast Summary

```text
Minimum prediction:  $813.71
Maximum prediction:  $2,481.14
Mean prediction:     $991.50
Median prediction:   $837.84
```

Prediction validation:

```text
Non-positive predictions: 0
Missing predictions:      0
```

The generated predictions are stored in:

```text
data/december-chart-inputs.csv
```

---

# 📊 December Forecast Visualization

The generated December forecast visualization is available at:

```text
scorer_results/candidate_december.png
```

The chart provides a visual representation of the predicted daily freight-rate pattern throughout December 2025.

---

# 🌎 Geographic Modeling

Geographic information is incorporated using pickup and delivery coordinates.

The project calculates:

```text
lat_diff
lon_diff
geographic_distance
```

The Haversine formula provides an approximate straight-line geographic distance between the two locations.

This information is combined with the reported shipment distance to derive:

```text
log_route_efficiency
```

This allows the model to represent both geographic separation and the actual reported route distance.

---

# 📦 Saved Artifacts

Reusable machine learning artifacts are stored under:

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

These artifacts allow the preprocessing and final model to be reused without rebuilding the entire pipeline from raw data.

---

# 🧩 Source Modules

The reusable source-code components are organized under:

```text
src/
```

### `features.py`

Contains feature-engineering logic used to construct derived modeling features.

### `preprocessing.py`

Contains preprocessing-related logic for transforming raw and engineered features into model-ready representations.

### `model.py`

Contains model-related functionality.

### `prediction.py`

Contains prediction-generation functionality used by the project pipeline.

This separation supports a more modular and maintainable project structure beyond the exploratory notebooks.

---

# 🔁 Reproducibility

The project follows a sequential workflow:

```text
01 → Data Exploration

02 → Feature Engineering & Preprocessing

03 → Modeling & Evaluation

04 → Final Model Training & Prediction
```

The workflow can therefore be inspected from raw data exploration through final prediction generation.

The trained preprocessing and model artifacts are persisted using `joblib`.

---

# ▶️ How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/zeinab-nasser/freight-rate-prediction.git
```

```bash
cd freight-rate-prediction
```

---

## 2. Create a Python Environment

A virtual environment can be created with:

```bash
python -m venv spotter-ml-env
```

Activate it on Windows:

```bash
spotter-ml-env\Scripts\activate
```

---

## 3. Install Dependencies

Install the project dependencies using:

```bash
pip install -r requirements.txt
```

---

## 4. Launch Jupyter

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

# 🧪 Output Validation

The project includes a dedicated scoring script:

```text
score.py
```

The generated validation file can be checked against the expected submission structure.

The final validation output follows the required schema:

```text
load_id
predicted_rate
```

and contains:

```text
12,000 predictions
```

---

# 💡 Key Technical Decisions

## Why Linear Regression?

Several regression approaches were evaluated on the same validation split.

Linear Regression achieved the lowest MAE:

```text
140.61
```

and was therefore selected for the final training stage based on the project's primary error metric.

---

## Why Feature Engineering?

Freight pricing depends on multiple interacting characteristics.

The project therefore provides the model with explicit representations of:

* Route geometry
* Geographic distance
* Shipment distance
* Shipment weight
* Weight per mile
* Temporal information
* Pickup and delivery locations
* Equipment type
* Market conditions
* Quote signals

---

## Why a Separate Final Training Stage?

Model comparison was performed using a validation split from the development data.

After selecting the model, the final Linear Regression model was retrained using all:

```text
48,000
```

development observations.

This allows the final model to use the complete labeled development dataset before generating predictions for the separate assessment validation set.

---

# ⚠️ Limitations & Future Improvements

Several areas could be explored in future iterations:

* Hyperparameter optimization for tree-based models
* Additional gradient-boosting algorithms
* Cross-validation for more robust model comparison
* More advanced route-level features
* Historical route statistics
* Route-specific pricing features
* Seasonal and holiday indicators
* Improved treatment of extreme target values
* Prediction intervals and uncertainty estimation
* Model monitoring
* Automated inference pipelines
* API deployment for real-time predictions

---

# 📌 Final Results

The completed project provides:

```text
✓ Exploratory data analysis

✓ Domain-informed feature engineering

✓ Geographic feature engineering

✓ Temporal feature engineering

✓ Reproducible preprocessing pipeline

✓ 148 processed model features

✓ Evaluation of multiple regression approaches

✓ Ensemble experimentation

✓ Validation-based model selection

✓ Final Linear Regression model

✓ 48,000-sample final training dataset

✓ 12,000 validation predictions

✓ Validated submission file

✓ December 2025 daily forecast

✓ December forecast visualization

✓ Saved preprocessing artifacts

✓ Saved trained model

✓ Modular source-code structure
```

---

# 👩‍💻 Author

**Zeinab Nasser**

Machine Learning / AI Engineer

GitHub:

https://github.com/zeinab-nasser

---

# 📚 Notebooks

| Notebook                                     | Purpose                                        |
| -------------------------------------------- | ---------------------------------------------- |
| `01_data_exploration.ipynb`                  | Data exploration and data-quality analysis     |
| `02_feature_engineering_preprocessing.ipynb` | Feature engineering and preprocessing          |
| `03_modeling_evaluation.ipynb`               | Model training, comparison, and evaluation     |
| `04_final_model.ipynb`                       | Final model training and prediction generation |

---

# 📁 Main Outputs

| Output                                     | Purpose                                       |
| ------------------------------------------ | --------------------------------------------- |
| `validation_predictions.csv`               | Final predictions for 12,000 validation loads |
| `data/december-chart-inputs.csv`           | December 2025 daily predictions               |
| `artifacts/final_linear_regression.joblib` | Trained final model                           |
| `artifacts/final_preprocessor.joblib`      | Fitted final preprocessing pipeline           |
| `scorer_results/candidate_december.png`    | December forecast visualization               |

---

## 🚀 End-to-End Summary

```text
Raw Freight Data
       ↓
Exploration
       ↓
Feature Engineering
       ↓
ColumnTransformer
       ↓
148 Processed Features
       ↓
Model Comparison
       ↓
Linear Regression Selected
       ↓
Full Development Training
       ↓
Final Model
       ├───────────────┐
       ↓               ↓
12,000 Validation   December 2025
Predictions         Forecast
       ↓               ↓
Submission File     Visualization
```

**Project repository:**
https://github.com/zeinab-nasser/freight-rate-prediction
