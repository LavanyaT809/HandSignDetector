import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data_dir = '../data'
X, y = [], []

for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    for file in os.listdir(label_dir):
        if file.endswith('.npy'):
            X.append(np.load(os.path.join(label_dir, file)))
            y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))

os.makedirs('../models', exist_ok=True)
joblib.dump(clf, '../models/hand_sign_rf.pkl')
