
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report#Loading DataSet

Datasheet = pd.read_csv(r'Cardio.csv')
#X = Datasheet.iloc[:, :-1].values  # All columns except the last one
X = Datasheet[['age','restingBP','serumcholestrol','fastingbloodsugar','restingrelectro','noofmajorvessels']]
Y = Datasheet.iloc[:, -1].values   # The last column

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=80)# Standardize the features

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)# Train the KNN model

knn = KNeighborsClassifier(n_neighbors=5)  # You can choose different values for k
knn.fit(X_train, Y_train)# Make predictions on the test set
y_pred = knn.predict(X_test)# Evaluate the model
accuracy = accuracy_score(Y_test, y_pred)
conf_matrix = confusion_matrix(Y_test, y_pred)
class_report = classification_report(Y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

import joblib
joblib.dump(knn, "knn_model.pkl")

joblib.dump(scaler, "scaler.pkl")