# 1. Core Libraries
import pandas as pd
import numpy as np
import pickle
import warnings

# 2. Preprocessing & Modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# 3. Ignore warnings
warnings.filterwarnings('ignore')


# Load and Preprocess Dataset
df = pd.read_csv("D:/University/Fundmentals of Data Science/Practical/Data/diabetes_prediction_dataset.csv")

# Separate features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['gender', 'smoking_history'], drop_first=False)

# Define expected columns for model consistency
required_columns = [
    'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
    'gender_Female', 'gender_Male', 'gender_Other',
    'smoking_history_current', 'smoking_history_former', 
    'smoking_history_never', 'smoking_history_not known'
]

# Ensure all required columns are present
for col in required_columns:
    X[col] = X.get(col, 0)

# Reorder columns
X = X[required_columns]

# Train/Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=200,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Save Model Artifacts
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'required_columns': required_columns
    }, f)

# Prediction Function
def create_test_case(age, hypertension, heart_disease, bmi, hba1c, glucose, 
                     gender, smoking_history):
    case = {col: 0 for col in required_columns}
    case.update({
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': glucose,
        f'gender_{gender}': 1,
        f'smoking_history_{smoking_history}': 1
    })
    return [case[col] for col in required_columns]

def predict_case(case):
    scaled = scaler.transform([case])
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]
    return prediction, probability

# Example Predictions
healthy_male = create_test_case(
    age=25, hypertension=0, heart_disease=0, bmi=22.0, 
    hba1c=4.8, glucose=85, gender='Male', smoking_history='never'
)

diabetic_female = create_test_case(
    age=60, hypertension=1, heart_disease=1, bmi=30.0,
    hba1c=8.5, glucose=200, gender='Female', smoking_history='current'
)

healthy_pred, healthy_proba = predict_case(healthy_male)
diabetic_pred, diabetic_proba = predict_case(diabetic_female)

# Display Results
print("\nTest Case Results:")
print(f"Healthy Male (expected 0): Prediction = {healthy_pred}, Probability = {healthy_proba:.2f}")
print(f"Diabetic Female (expected 1): Prediction = {diabetic_pred}, Probability = {diabetic_proba:.2f}")
