# ===========================================
# Student Academic Performance Prediction
# Model Training Script
# ===========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ===========================================
# Load Dataset
# ===========================================

print("="*60)
print("Loading Dataset...")
print("="*60)

df = pd.read_csv("Final_Project_Dataset.csv")

print(df.head())

print("\nDataset Shape :", df.shape)

print("\nColumns :")

print(df.columns.tolist())

# ===========================================
# Check Missing Values
# ===========================================

print("\nMissing Values")

print(df.isnull().sum())

# ===========================================
# Remove Duplicates
# ===========================================

duplicates = df.duplicated().sum()

print("\nDuplicate Rows :", duplicates)

df = df.drop_duplicates()

# ===========================================
# Encode Categorical Columns
# ===========================================

print("\nEncoding Categorical Columns...")

label_encoders = {}

categorical_columns = [
    "gender",
    "internet_quality",
    "motivation_level",
    "parental_education",
    "family_income",
    "part_time_job"
]

for col in categorical_columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    label_encoders[col] = le

print("Encoding Completed")

# ===========================================
# Save Label Encoders
# ===========================================

joblib.dump(
    label_encoders,
    "label_encoders.pkl"
)

print("label_encoders.pkl saved")

# ===========================================
# Features and Target
# ===========================================

X = df.drop("exam_score", axis=1)

y = df["exam_score"]

# ===========================================
# Feature Scaling
# ===========================================

print("\nScaling Features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("scaler.pkl saved")

# ===========================================
# Train Test Split
# ===========================================

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,

    y,

    test_size=0.20,

    random_state=42

)

print("\nTraining Samples :", len(X_train))

print("Testing Samples :", len(X_test))










# ===========================================
# Train Models
# ===========================================

print("\n" + "="*60)
print("Training Machine Learning Models")
print("="*60)

# -------------------------------
# Linear Regression
# -------------------------------

print("\nTraining Linear Regression...")

lr = LinearRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("Linear Regression Completed")


# -------------------------------
# Decision Tree
# -------------------------------

print("\nTraining Decision Tree...")

dt = DecisionTreeRegressor(
    random_state=42
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("Decision Tree Completed")


# -------------------------------
# Random Forest
# -------------------------------

print("\nTraining Random Forest...")

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=15
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Random Forest Completed")


# -------------------------------
# XGBoost (Optional)
# -------------------------------

xgb_available = False

try:

    from xgboost import XGBRegressor

    print("\nTraining XGBoost...")

    xgb = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    xgb.fit(X_train, y_train)

    xgb_pred = xgb.predict(X_test)

    xgb_available = True

    print("XGBoost Completed")

except ImportError:

    print("\nXGBoost not installed.")
    print("Skipping XGBoost...")


# ===========================================
# Evaluation Function
# ===========================================

def evaluate_model(model_name, y_true, prediction):

    mae = mean_absolute_error(
        y_true,
        prediction
    )

    mse = mean_squared_error(
        y_true,
        prediction
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        prediction
    )

    return {

        "Model": model_name,

        "MAE": round(mae,3),

        "MSE": round(mse,3),

        "RMSE": round(rmse,3),

        "R2 Score": round(r2,3)

    }


# ===========================================
# Evaluate Models
# ===========================================

results = []

results.append(
    evaluate_model(
        "Linear Regression",
        y_test,
        lr_pred
    )
)

results.append(
    evaluate_model(
        "Decision Tree",
        y_test,
        dt_pred
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        rf_pred
    )
)

if xgb_available:

    results.append(
        evaluate_model(
            "XGBoost",
            y_test,
            xgb_pred
        )
    )

results_df = pd.DataFrame(results)

print("\n")
print("="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(results_df)











# ===========================================
# Select Best Model
# ===========================================

print("\n" + "="*60)
print("SELECTING BEST MODEL")
print("="*60)

best_index = results_df["R2 Score"].idxmax()

best_model_name = results_df.loc[best_index, "Model"]

print(f"\nBest Model : {best_model_name}")

# ===========================================
# Save Best Model
# ===========================================

if best_model_name == "Linear Regression":

    best_model = lr

elif best_model_name == "Decision Tree":

    best_model = dt

elif best_model_name == "Random Forest":

    best_model = rf

else:

    best_model = xgb

joblib.dump(best_model, "best_model.pkl")

print("best_model.pkl saved successfully")

# ===========================================
# Save Results CSV
# ===========================================

results_df.to_csv(
    "model_results.csv",
    index=False
)

print("model_results.csv saved")

# ===========================================
# Feature Importance
# ===========================================

import matplotlib.pyplot as plt

feature_names = X.columns

if best_model_name in ["Random Forest", "Decision Tree"]:

    importance = best_model.feature_importances_

elif best_model_name == "XGBoost":

    importance = best_model.feature_importances_

else:

    importance = np.abs(best_model.coef_)

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)

print("\nTop Important Features")

print(importance_df)

importance_df.to_csv(

    "feature_importance.csv",

    index=False

)

# ===========================================
# Feature Importance Plot
# ===========================================

plt.figure(figsize=(10,6))

plt.barh(

    importance_df["Feature"],

    importance_df["Importance"]

)

plt.xlabel("Importance")

plt.ylabel("Features")

plt.title("Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()

print("feature_importance.png saved")

# ===========================================
# Final Accuracy Report
# ===========================================

print("\n" + "="*60)
print("FINAL MODEL RESULTS")
print("="*60)

print(results_df)

print("\nBest Model")

print(best_model_name)

best_row = results_df.iloc[best_index]

print(f"R2 Score : {best_row['R2 Score']}")

print(f"MAE      : {best_row['MAE']}")

print(f"RMSE     : {best_row['RMSE']}")

# ===========================================
# Prediction Example
# ===========================================

print("\nRunning Sample Prediction...")

sample_prediction = best_model.predict(

    X_test[:1]

)

print(

    f"Predicted Exam Score : {sample_prediction[0]:.2f}"

)

print(

    f"Actual Exam Score    : {y_test.iloc[0]}"

)

# ===========================================
# Success Message
# ===========================================

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("="*60)

print("""
Generated Files

✔ best_model.pkl
✔ scaler.pkl
✔ label_encoders.pkl
✔ model_results.csv
✔ feature_importance.csv
✔ feature_importance.png

Your Machine Learning Model is Ready.
""")