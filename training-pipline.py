import pickle
import argparse
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

data = pickle.loads(open('encodings.pkl', "rb").read())

# train a classification model on these embeddings
# use the model to make predictions on the test data

X = data['encodings']
y_raw = data['names']

le = LabelEncoder()
y = le.fit_transform(y_raw)

print(y)

# model = SVC(kernel='linear', probability=True)
# model.fit(X, y)


# accuracy = model.score(X, y)




