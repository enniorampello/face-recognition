import pickle
import argparse
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier



def train(storage_client, bucket_name, embeddings_file):

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(embeddings_file)
    blob.download_to_filename(embeddings_file)

    data = pickle.loads(open(embeddings_file, "rb").read())

    # train a classification model on these embeddings
    # use the model to make predictions on the test data

    X = data['encodings']
    y_raw = data['names']

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(le.classes_)
    # save the labels in a file 
    f = open('labels.pkl', "wb")
    f.write(pickle.dumps(le.classes_))
    f.close()

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    accuracy = model.score(X, y)
    print(f'Accuracy: {accuracy}')

    #save the model to disk
    f = open('model.pkl', "wb")
    f.write(pickle.dumps(model))
    f.close()


if __name__ == '__main__':
    train()