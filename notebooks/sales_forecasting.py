# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# Load the datasets
features_df = pd.read_csv(r'C:\Users\lenovo\Downloads\Sales forcast\data\Features data set.csv')
sales_df = pd.read_csv(r'C:\Users\lenovo\Downloads\Sales forcast\data\sales data-set.csv')
stores_df = pd.read_csv(r'C:\Users\lenovo\Downloads\Sales forcast\data\stores data-set.csv')

# Check the shape of the data
print("Features data shape:", features_df.shape)
print("Sales data shape:", sales_df.shape)
print("Stores data shape:", stores_df.shape)

# Display first few rows of each dataset
print("\nFeatures Data:")
print(features_df.head())

print("\nSales Data:")
print(sales_df.head())

print("\nStores Data:")
print(stores_df.head())
# Convert 'Date' columns to datetime format
features_df['Date'] = pd.to_datetime(features_df['Date'], dayfirst=True)
sales_df['Date'] = pd.to_datetime(sales_df['Date'], dayfirst=True)
stores_df['Store'] = stores_df['Store'].astype(int)

# Merge sales and features data on Store and Date
merged_df = pd.merge(sales_df, features_df, how='left', on=['Store', 'Date'])

# Merge with stores data
merged_df = pd.merge(merged_df, stores_df, how='left', on='Store')

# Check the merged data
print("\nMerged Data:")
print(merged_df.head())
print("\nMerged Data Shape:", merged_df.shape)
# --------------------------
# Step 3: Exploratory Data Analysis (EDA)
# --------------------------

print("\nMissing Values:")
print(merged_df.isnull().sum())

print("\nData Types:")
print(merged_df.dtypes)

# Correlation heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(merged_df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Sales trend over time
plt.figure(figsize=(12, 6))
merged_df.groupby('Date')['Weekly_Sales'].sum().plot()
plt.title('Total Weekly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Sales by store type
plt.figure(figsize=(8, 5))
sns.barplot(x='Type', y='Weekly_Sales', data=merged_df)
plt.title('Average Weekly Sales by Store Type')
plt.show()

# Sales by holiday
plt.figure(figsize=(8, 5))
sns.boxplot(x='IsHoliday_x', y='Weekly_Sales', data=merged_df)
plt.title('Sales Distribution During Holidays vs Non-Holidays')
plt.show()

# Fill NaN in Markdown columns with 0
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
merged_df[markdown_cols] = merged_df[markdown_cols].fillna(0)

print("\nMissing values after filling:")
print(merged_df.isnull().sum())

# Extract Year, Month, Week, DayOfWeek from Date
merged_df['Year'] = merged_df['Date'].dt.year
merged_df['Month'] = merged_df['Date'].dt.month
merged_df['Week'] = merged_df['Date'].dt.isocalendar().week.astype(int)
merged_df['Day'] = merged_df['Date'].dt.day
merged_df['DayOfWeek'] = merged_df['Date'].dt.dayofweek

from sklearn.preprocessing import LabelEncoder

# Encode 'Type'
le = LabelEncoder()
merged_df['Type'] = le.fit_transform(merged_df['Type'])

# Convert boolean to integer
merged_df['IsHoliday_x'] = merged_df['IsHoliday_x'].astype(int)
merged_df['IsHoliday_y'] = merged_df['IsHoliday_y'].astype(int)

merged_df.drop(['Date', 'IsHoliday_y'], axis=1, inplace=True)

print(merged_df.head())
print("\nData shape:", merged_df.shape)
print("\nData types:\n", merged_df.dtypes)

from sklearn.model_selection import train_test_split

# Define features and target
X = merged_df.drop('Weekly_Sales', axis=1)
y = merged_df['Weekly_Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Initialize the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'Random Forest RMSE: {rmse:.2f}')
print(f'Random Forest R²: {r2:.4f}')

# ✅ Feature Importance Plot
import matplotlib.pyplot as plt
import seaborn as sns

feature_importance = rf_model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance')
plt.show()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ✅ Hyperparameter Tuning
param_distributions = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_distributions,
    n_iter=5,
    cv=2,
    scoring='r2',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
best_model = random_search.best_estimator_

# ✅ Evaluate Best Model
y_pred_best = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_best)
r2 = r2_score(y_test, y_pred_best)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R2 Score: {r2:.2f}')

# ✅ Feature Importance
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# ✅ Save Model
joblib.dump(best_model, 'final_sales_prediction_model.pkl')
print("Model saved successfully as 'final_sales_prediction_model.pkl'")

# ✅ Optional: Summary for Portfolio
summary = """
### Project Summary

- **Objective**: Predict weekly sales for Walmart stores based on historical sales data, promotions, and store information.
- **Data Cleaning**: Handled missing values, date features extracted, categorical encoding.
- **Model Used**: RandomForestRegressor with hyperparameter tuning (RandomizedSearchCV).
- **Performance**: Achieved R2 Score of {:.2f}, MSE of {:.2f}.
- **Feature Importance**: Top features include {}, {}, and {}.
- **Model Exported**: Saved as 'final_sales_prediction_model.pkl' for future use.
""".format(
    r2,
    mse,
    importance_df.iloc[0]['Feature'],
    importance_df.iloc[1]['Feature'],
    importance_df.iloc[2]['Feature']
)

print(summary)

# ✅ Visualizations
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.3)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance')
plt.show()









