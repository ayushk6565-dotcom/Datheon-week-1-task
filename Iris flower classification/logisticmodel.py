import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

data = pd.read_csv("Iris flower classification/iris.csv")
df = pd.DataFrame(data)
print(df.isnull().sum())
X= df.drop("species",axis=1)
y = df["species"]
scaler = StandardScaler()
X = scaler.fit_transform(X)
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,train_size=0.8)
model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
prediction = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, prediction))

print("Precision:", precision_score(y_test, prediction, average="weighted"))

print("Recall:", recall_score(y_test, prediction, average="weighted"))

print("F1 Score:", f1_score(y_test, prediction, average="weighted"))

print(classification_report(y_test, prediction))
cm = confusion_matrix(y_test, prediction)
print(cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()