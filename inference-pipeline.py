# import the necessary packages
import face_recognition
import pickle
import argparse
import cv2
from sklearn.neighbors import KNeighborsClassifier
import gradio_app as gr
import numpy as np


def inf(image):
    # input_image = 'multiple.jpg'
    # encodings_file = 'encodings.pkl'
    detection_method = 'hog'
    classifier_model_file = 'model.pkl'
    labels_file = 'labels.pkl'
    # data = pickle.loads(open(encodings_file, "rb").read())

    # image = cv2.imread(input_image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb_image, model=detection_method)
    encodings = face_recognition.face_encodings(rgb_image, boxes)
    print(f'Found {len(boxes)} faces')

    # initialize the list of names for each face detected
    names = []

    # # load the model from disk
    model = pickle.loads(open(classifier_model_file, "rb").read())
    # predict on the new encodings and display the probabilities
    y_pred = model.predict_proba(encodings)
    # print(y_pred)
    max_y = y_pred.max(axis=1)
    print(max_y)
    labels = pickle.loads(open(labels_file, "rb").read())
    for id, m in enumerate(max_y):
        
        if m >= 0.8:
            names.append(labels[y_pred.argmax(axis=1)[id]])
        else:
            names.append('Unknown')


    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    # save the resulting image
    # cv2.imwrite('output.jpg', image)
    return image


demo = gr.Interface(
    inf, 
    gr.Image(source="webcam", streaming=True), 
    "image",
    live=True
)
demo.launch()

