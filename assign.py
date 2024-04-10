import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

titanic_df = pd.read_csv("titanic_dataset.csv")

features = ['Class', 'Gender', 'Age', 'Siblings', 'Parents', 'Fare', 'Port']
X = titanic_df[features]
y = titanic_df['Survived']
X_encoded = pd.get_dummies(X)
X_encoded.fillna(X_encoded.mean(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)
log_reg_preds = log_reg_model.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_preds)
print("Logistic Regression Accuracy:", log_reg_accuracy)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_preds)
print("SVM Accuracy:", svm_accuracy)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_preds)
print("Decision Tree Accuracy:", dt_accuracy)
