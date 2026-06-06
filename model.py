import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data.csv')


print("Shape:", df.shape)     
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:", df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())
print("\nData types:")
print(df.dtypes)

df.fillna(df.median(numeric_only=True), inplace=True)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

print("\nAfter cleaning — any nulls left?", df.isnull().sum().sum())
print("All columns now numeric:", df.dtypes.unique())

TARGET = 'loan_status'

X = df.drop(columns=[TARGET])   
y = df[TARGET]                   

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model trained successfully!")
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

y_pred = model.predict(X_test)

print("\n=== MODEL RESULTS ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()


feat_imp = pd.Series(model.feature_importances_,
                     index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
feat_imp[:10].plot(kind='bar', color='steelblue')
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
import joblib
joblib.dump(model, 'credit_scoring_model.pkl')
print("Model saved!")