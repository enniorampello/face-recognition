import time
import imutils
from imutils import paths
import face_recognition
import pickle
import cv2
import os
import argparse


dataset = './dataset/' # path to input dataset directory 
encodings_file = './encodings.pkl' #path to output encodings file
detection_method = 'hog'

imagePaths = list(paths.list_images(dataset))
knownEncodings = []
knownNames = []
s = time.time()

for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2] 
    print(f"processing image [{name}] {i+1}/{len(imagePaths)}")
    images = []
    image = cv2.imread(imagePath)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(rgb_image)
    # rotate the image by 10 degrees and add it to the list of images
    images.append(imutils.rotate_bound(rgb_image, 10))
    # rotate the image by -10 degrees and add it to the list of images
    images.append(imutils.rotate_bound(rgb_image, -10))
    # rotate the image by 30 degrees and add it to the list of images
    images.append(imutils.rotate_bound(rgb_image, 30))
    # rotate the image by -30 degrees and add it to the list of images
    images.append(imutils.rotate_bound(rgb_image, -30))

    # flip the image horizontally and add it to the list of images
    images.append(cv2.flip(rgb_image, 1))

    #increase the brightness of the image and add it to the list of images
    images.append(cv2.convertScaleAbs(rgb_image, alpha=1.5, beta=0))
    #decrease the brightness of the image and add it to the list of images
    images.append(cv2.convertScaleAbs(rgb_image, alpha=0.5, beta=0))

    print(f'created {len(images)} images for {name}')


    # detect the (x,y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    # we are assuming the the boxes of faces are the SAME FACE or SAME PERSON
    # boxes = face_recognition.face_locations(rgb_image, model=detection_method)
    # print(f"Found {len(boxes)} faces in image")

    for l in images:
        boxes = [(0, l.shape[1], l.shape[0], 0)]
        encodings = face_recognition.face_encodings(l, boxes)

        # creating the training set
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

e = time.time()
print(f"Encoding dataset took: {(e-s)} seconds")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(encodings_file, "wb")
f.write(pickle.dumps(data))
f.close()

# replace the above few lines with the script to upload this to hopsworks