# train_model.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
df = pd.read_excel("churn.xlsx")

# 2. Features & Target
target = "Exited"   # agar target column ka naam kuch aur hai to yaha badalna
X = df.drop(columns=[target, "CustomerId", "Surname"], errors="ignore")
y = df[target]

# 3. Encode categorical features
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# 6. Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

# 7. Save model
joblib.dump(model, "churn_model.pkl")
print("✅ Model saved as churn_model.pkl")