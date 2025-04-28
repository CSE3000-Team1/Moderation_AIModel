#import necessary libraries
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# load the preprocessed data
with open('preprocessor.pkl', 'rb') as f:
    X, y = pickle.load(f)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate the model
y_pred = model.predict(X_test)

# print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-Score:", f1_score(y_test, y_pred, average='weighted'))

# plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)  # confusion matrix
sns.heatmap(cm, annot=True, fmt='d')   # plot the confusion matrix
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# save the trained model
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

#print confirmation message
print("Model training complete. Model saved to 'trained_model.pkl'.")