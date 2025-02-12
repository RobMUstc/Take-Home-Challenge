import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from feature_extraction import select_features

vectoriser = TfidfVectorizer()

# Store preprocessed data and chosen features
train_df, test_df, features = select_features(vectoriser)

# Prepare feature matrix and target vector
X = train_df[features]
y = train_df['label_5way']

# Split feature matrix and target vector into actual training and validation subsamples (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {'contradictory': class_weights[0], 'correct': class_weights[1], 'irrelevant': class_weights[2],
                     'non_domain': class_weights[3], 'partially_correct_incomplete': class_weights[4]}

# Train the classification model
rf = RandomForestClassifier(class_weight=class_weight_dict)

rf.fit(X_train, y_train)

# Evaluate the model on the training set
y_train_pred = rf.predict(X_train)
print(f"\nTraining Accuracy: {accuracy_score(y_train, y_train_pred):.2f}")
print("Classification Report on Training Set:")
print(classification_report(y_train, y_train_pred))

# Calculate and visualise the feature importance scores from the trained model
feature_importances = rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)


# Evaluate the model on the validation set
y_val_pred = rf.predict(X_val)
print(f"\nValidation Accuracy: {accuracy_score(y_val, y_val_pred):.2f}")
print("Classification Report on Validation Set:")
print(classification_report(y_val, y_val_pred))

# Tune hyperparameters (code redacted, refer to misc_work script)



# RUN MODEL ON THE TEST DATA

# Prepare the feature matrix (X_test) for prediction
testUA_df = test_df[test_df['test_set'] == 'unseen-answers']
testUQ_df = test_df[test_df['test_set'] == 'unseen-questions']

X_testUA = testUA_df[features]
X_testUQ = testUQ_df[features]

# Predict the grades for the test dataset
predicted_gradesUA = rf.predict(X_testUA)
predicted_gradesUQ = rf.predict(X_testUQ)


# ERROR ANALYSIS

# Predicted grades, classification report and confusion matrix for unseen answers
y_testUA_pred = rf.predict(X_testUA)
y_testUA = testUA_df['label_5way']
print(f"\nTest Accuracy: {accuracy_score(y_testUA, y_testUA_pred):.2f}")
print("Classification Report on Unseen Answers Test Set:")
print(classification_report(y_testUA, y_testUA_pred))
cmUA = confusion_matrix(y_testUA, y_testUA_pred, labels=['correct', 'partially_correct_incomplete', 'irrelevant',
                                                         'contradictory', 'non_domain'])
dispUA = ConfusionMatrixDisplay(confusion_matrix=cmUA, display_labels=['correct', 'partially_correct_incomplete',
                                                                   'irrelevant', 'contradictory', 'non_domain'])
dispUA.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Unseen Answers Test Set:")
plt.show()

# Predicted grades, classification report and confusion matrix for unseen questions
y_testUQ_pred = rf.predict(X_testUQ)
y_testUQ = testUQ_df['label_5way']
print(f"\nTest Accuracy: {accuracy_score(y_testUQ, y_testUQ_pred):.2f}")
print("Classification Report on Unseen Questions Test Set:")
print(classification_report(y_testUQ, y_testUQ_pred))
cmUQ = confusion_matrix(y_testUQ, y_testUQ_pred, labels=['correct', 'partially_correct_incomplete', 'irrelevant',
                                                         'contradictory', 'non_domain'])
dispUQ = ConfusionMatrixDisplay(confusion_matrix=cmUQ, display_labels=['correct', 'partially_correct_incomplete',
                                                                     'irrelevant', 'contradictory', 'non_domain'])
dispUQ.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Unseen Questions Test Data")
plt.show()
