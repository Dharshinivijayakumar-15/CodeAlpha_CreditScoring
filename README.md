#  Credit Scoring Model

> Predicting loan defaulters using Machine Learning 

---

##  Objective
Predict whether a person will **default on a loan** based on their financial history and personal data.

---

##  How It Works
- Financial dataset loaded and cleaned using **pandas**
- Categorical columns encoded using **LabelEncoder**
- Data split into 80% training and 20% testing
- **Random Forest Classifier** trained on 32,000+ records
- Model evaluated using accuracy, precision, recall and confusion matrix

---

##  Tech Stack
- Python
- scikit-learn
- pandas
- matplotlib & seaborn
- Random Forest Classifier

---

##  Project Structure
```
credit_scoring_model/
├── model.py              → Full ML pipeline
├── data.csv              → Cleaned dataset
├── confusion_matrix.png  → Model evaluation chart
└── feature_importance.png → Top features chart
```

---

##  How to Run
```
pip install pandas scikit-learn matplotlib seaborn
python model.py
```

---

##  Results
- **Accuracy: 92.97%**
- **Precision: 96%**
- **Recall: 71%**
