# 🛒 Walmart Sales Forecasting Project

## 📌 Objective
Predict weekly sales for Walmart stores using historical sales data, promotions, and store information.

## 📊 Dataset
- **Train.csv** — Historical training data
- **Features.csv** — Additional data like promotions, fuel price, etc.
- **Stores.csv** — Store type and size information

## 🧹 Data Preparation
- Handled missing values
- Extracted date features (Year, Month, Week, Day)
- Encoded categorical variables
- Merged datasets for enriched features

## 🤖 Model Building
- Algorithm: **RandomForestRegressor**
- Hyperparameter Tuning: `RandomizedSearchCV`
- Final Parameters:
  - `n_estimators = 150`
  - `max_depth = 20`
  - `min_samples_split = 2`
  - `min_samples_leaf = 1`

## 🏆 Performance
- **R² Score:** 0.98
- **Mean Squared Error:** 12,098,694.38
- Model exported as: `final_sales_prediction_model.pkl`

## 🔍 Feature Importance
- Top Features:
  - Department
  - Store Size
  - Store ID

## 📈 Visualizations
- Feature importance bar plot
- Model performance metrics

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Akashkc7/walmart-sales-forecasting.git
   cd walmart-sales-forecasting



