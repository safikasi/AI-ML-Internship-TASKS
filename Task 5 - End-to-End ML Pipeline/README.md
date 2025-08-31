# Task 5: End-to-End ML Pipeline with Scikit-learn Pipeline API

## 📌 Objective
Build a reusable and **production-ready machine learning pipeline** for predicting **customer churn** using the **Telco Churn Dataset**.  

The pipeline leverages **Scikit-learn's Pipeline API** for streamlined preprocessing, training, hyperparameter tuning, and deployment.

---

## 📊 Dataset
We use the **Telco Customer Churn Dataset**.  
The dataset typically includes features such as:
- **Customer demographics** (gender, age, senior citizen, etc.)
- **Account information** (tenure, contract type, billing method, etc.)
- **Service details** (internet service, phone service, streaming, etc.)
- **Target variable**: `Churn` (Yes/No → converted to 1/0)

---

## 🛠️ Steps Implemented

### 1. Data Preprocessing
- Handle missing values
- Encode categorical variables using **OneHotEncoder**
- Scale numerical features using **StandardScaler**
- Use **ColumnTransformer** inside a `Pipeline`

### 2. Model Training
- Implemented **Logistic Regression** and **Random Forest** within the pipeline
- Ensured that preprocessing + model training are combined in a **single reusable pipeline**

### 3. Hyperparameter Tuning
- Applied **GridSearchCV** for:
  - Logistic Regression (regularization strength, penalty)
  - Random Forest (number of trees, max depth, etc.)
- Selected the best model based on **cross-validation performance**

### 4. Evaluation
- Metrics used:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**

### 5. Exporting the Model
- Used **joblib** to export the **trained pipeline**
- The saved pipeline includes preprocessing + model → ready for production

## 📦 Skills Gained

✅ ML pipeline construction with Scikit-learn\
✅ Data preprocessing with Pipeline + ColumnTransformer\
✅ Hyperparameter tuning using GridSearchCV\
✅ Model export & reusability with joblib\
✅ Production-readiness practices