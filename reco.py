#!/usr/bin/env python3

# https://krasserm.github.io/2018/02/07/deep-face-recognition/
# http://nbviewer.jupyter.org/github/krasserm/face-recognition/blob/master/face-recognition.ipynb?flush_cache=true

import os

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from pytoolbox import filesystem
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from pytoolbox.ai.vision import utils
from pytoolbox.ai.vision.face import detect, recognize

IMAGES_DIRECTORY = os.path.join(os.path.dirname(__file__), 'images')


def main_demo():

    # Initialize
    detector = detect.DlibFaceDetector()
    recognizer = recognize.load_nn4_small2_model()
    identities = load_identities(IMAGES_DIRECTORY)
    vectors = np.zeros((identities.shape[0], 128))
    for counter, identity in enumerate(identities):
        box, face = detector.extract_largest_face(identity.load())
        vectors[counter] = recognizer.predict(np.expand_dims(utils.normalize_rgb(face), axis=0))[0]

    # Train
    targets = np.array([i.name for i in identities])
    encoder = LabelEncoder()
    encoder.fit(targets)

    # Numerical encoding of identities
    y = encoder.transform(targets)

    # Train with only 1/4 of the dataset, then test with 3/4
    test_idx = np.arange(identities.shape[0]) % 4 != 0
    train_idx = np.arange(identities.shape[0]) % 4 == 0

    X_train = vectors[train_idx]
    X_test = vectors[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    svc = LinearSVC()

    knn.fit(X_train, y_train)
    svc.fit(X_train, y_train)

    # Predict
    acc_knn = accuracy_score(y_test, knn.predict(X_test))
    acc_svc = accuracy_score(y_test, svc.predict(X_test))
    print(f'KNN accuracy = {acc_knn}, SVM accuracy = {acc_svc}')


def main_show():
    detector = detect.DlibFaceDetector()

    for identity in load_identities(IMAGES_DIRECTORY):
        image = identity.load()
        plt.imshow(image)
        for box in detector.get_all_faces_bounding_boxes(image):
            print(identity.path, 'Detected face at coordinates', box)
            plt.gca().add_patch(patches.Rectangle(
                (box.left(), box.top()), box.width(), box.height(), fill=False, color='red'))
        plt.show()


class Identity(object):

    __slots__ = ('name', 'path')

    def __init__(self, name, path):
        self.name = name  # person name
        self.path = path  # image path

    def __repr__(self):
        return f"Identity(name='{self.name}', path='{self.path}')"

    def load(self):
        return utils.load_image(self.path)


def load_identities(path, patterns=('*.jpg', '*.jpeg'), **kwargs):
    identities = []
    for name in os.listdir(path):
        for filename in filesystem.find_recursive(os.path.join(path, name), patterns, **kwargs):
            identities.append(Identity(name, filename))
    return np.array(identities)


if __name__ == '__main__':
    main_demo()
