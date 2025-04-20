Titanic_Survival_Prediction
(🚢 Titanic Survival Prediction using Machine Learning)
This project aims to build a classification model that predicts whether a passenger survived the Titanic disaster based on various features.

🎯 Objective

To develop a machine learning pipeline that:
- Preprocesses the Titanic dataset
- Handles missing values
- Encodes categorical variables
- Normalizes numerical features
- Trains and evaluates a classification model
- Achieves strong performance on survival prediction
- Provides a reproducible, well-documented solution

📁 Dataset

The dataset used is publicly available on Kaggle:  
🔗 [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/brendan45774/test-file)

It consists of information about Titanic passengers including:
- PassengerId
- Name
- Age
- Sex
- Ticket
- Fare
- Cabin
- Pclass (Passenger class)
- SibSp (No. of siblings/spouses aboard)
- Parch (No. of parents/children aboard)
- Embarked (Port of embarkation)
- Survived (Target variable: 1 = Survived, 0 = Did not survive)

🧼 Data Preprocessing

🔍 1. Missing Value Handling
- `Age`: Filled with **median**
- `Embarked`: Filled with **mode**
- `Cabin`: Dropped due to high % of missing data

🔁 2. Dropped Columns
- `PassengerId`, `Ticket`, `Name`, `Cabin`

🔠 3. Encoding Categorical Variables
- `Sex` and `Embarked` encoded using **LabelEncoder**

📊 4. Feature Scaling
- Features scaled using **StandardScaler** to normalize distributions

🧠 Model Building

We chose a **Random Forest Classifier** due to its:
- Robustness to overfitting
- Ability to handle both numerical and categorical data
- Built-in feature importance evaluation

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
📈 Model Evaluation
The model performed exceptionally well on the test set:


Metric	Value
Accuracy	1.00
Precision	1.00
Recall	1.00
F1 Score	1.00
🧾 Classification Report

precision    recall  f1-score   support

           0       1.00      1.00      1.00        53
           1       1.00      1.00      1.00        31

    accuracy                           1.00        84
   macro avg       1.00      1.00      1.00        84
weighted avg       1.00      1.00      1.00        84

🧠 Note: While a perfect score is impressive, be cautious—it might indicate:

A very easy split of the dataset,

Potential data leakage, or

A small test set that doesn't generalize well.

💾 Files Included
Titanic_Survival_Prediction.ipynb: Google Colab notebook with full implementation

train.csv: Input dataset

titanic_rf_model.pkl: (Optional) Pickled trained model

README.md: Project documentation

🧪 How to Run This Project
Open Google Colab

Upload tested.csv to the notebook session

Run the notebook cells step-by-step

Modify/test with other models (e.g., Logistic Regression, XGBoost) for further improvements

Optional: Save and export the model for reuse

📌 Future Improvements
Hyperparameter tuning using GridSearchCV

Feature engineering (e.g., family size, title extraction from names)

Use of ensemble models and cross-validation

Deployment as a web app using Flask/Streamlit


📬 Contact
Project by: Kajal Pawar
📧 Email: kajal03.pawar@gmail.com.com
