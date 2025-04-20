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
Predictions were evaluated using standard classification metrics:


Metric	Value
Accuracy	~0.81
Precision	~0.79
Recall	~0.75
F1 Score	~0.77
🧾 Classification Report
markdown
Copy
Edit
              precision    recall  f1-score   support

           0       0.85      0.87      0.86       105
           1       0.76      0.73      0.75        74

    accuracy                           0.81       179
   macro avg       0.81      0.80      0.80       179
weighted avg       0.81      0.81      0.81       179

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
