import time
import imutils
from imutils import paths
import face_recognition
import pickle
import cv2
import os
import argparse

def augment_image(file_path):
    """
    Augment the image by rotating it by 10, -10, 30, -30 degrees, flipping it horizontally and increasing/decreasing the brightness

    params:
        file_path: path to the image file
    
    returns:
        list of 8 augmented images
    """
    image = cv2.imread(file_path)
    images = []
    
    # convert the input image from BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # add the original image to the list of images
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

    return images


def preprocess():

    dataset = './dataset' # path to input dataset directory
    encodings_file = './encodings.pkl' #path to output encodings file

    imagePaths = list(paths.list_images(dataset))
    known_encodings = []
    known_names = []
    s = time.time()
    print(imagePaths)
    for i, imagePath in enumerate(imagePaths):
        images = augment_image(imagePath)
        name = imagePath.split(os.path.sep)[-2] 
        print(f"processing image [{name}] {i+1}/{len(imagePaths)}")
    
        for l in images:
            boxes = face_recognition.face_locations(l, model='hog')
            encodings = face_recognition.face_encodings(l, boxes)

            # creating the training set
            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(name)

    e = time.time()
    print(f"Encoding dataset took: {(e-s)} seconds")
    data = {"encodings": known_encodings, "names": known_names}
    f = open(encodings_file, "wb")
    f.write(pickle.dumps(data))
    f.close()

    # replace the above few lines with the script to upload this to hopsworks

if __name__ == '__main__':
    preprocess()