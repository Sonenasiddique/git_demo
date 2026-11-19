import pandas as pd
import numpy as np

df = pd.read_csv("analysis.csv")

df = df.rename(columns={"Target":"Heart Disease"})
print(df.columns)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score , f1_score , classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

X = df.drop('Heart Disease',axis=1)
y = df['Heart Disease']

X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.20, random_state=42)

scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_Scaled  = scaler.fit_transform(X_test)

models ={
"Logistic_model":LogisticRegression(),
"Knn_model": KNeighborsClassifier(),
"Naive_bayes_model": GaussianNB(),
"Decision_tree_model":DecisionTreeClassifier(),
"Svm_model":SVC()
}
result =  []

for name,model in models.items():
    model.fit(X_train_scaled,y_train)
    y_pred = model.predict(X_test_Scaled)
    acc = accuracy_score(y_test,y_pred)
    f1_scores =  f1_score(y_test, y_pred)
    result.append({
        "model  ":name,
        "accuracy ": round(acc,4),
        "F1_score " : round(f1_scores, 4)
    })

print(result)


import joblib
joblib.dump(models['Knn_model'],'KNN_heart.pkl')
joblib.dump(scaler,'Scaler.pkl')
joblib.dump(X.columns.to_list(),'columns.pkl')