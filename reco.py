#!/usr/bin/env python3

# https://krasserm.github.io/2018/02/07/deep-face-recognition/
# http://nbviewer.jupyter.org/github/krasserm/face-recognition/blob/master/face-recognition.ipynb?flush_cache=true

import bz2, os
from urllib.request import urlopen

import cv2, numpy as np
from pytoolbox import filesystem
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

import face_reco
from face_reco.align import AlignDlib
from face_reco.model import create_model

face_reco_path = face_reco.__path__._path[0]

ALIGNMENT_DATA_FILENAME = os.path.join(os.path.dirname(__file__), 'landmarks.dat')


def main():

    # Initialize
    alignment = load_alignment_lib()
    model = load_model()
    identities = load_identities(os.path.join(face_path, 'images'))
    embedded = np.zeros((identities.shape[0], 128))
    for counter, identity in enumerate(identities):
        image = load_image(identity.path)
        image = align_image(alignment, image)
        # scale RGB values to interval [0,1]
        image = (image / 255).astype(np.float32)
        # obtain embedding vector for image
        embedded[counter] = model.predict(np.expand_dims(image, axis=0))[0]

    # Train
    targets = np.array([i.name for i in identities])
    encoder = LabelEncoder()
    encoder.fit(targets)

    # Numerical encoding of identities
    y = encoder.transform(targets)

    # Train with only 1/4 of the dataset, then test with 3/4
    test_idx = np.arange(identities.shape[0]) % 4 != 0
    train_idx = np.arange(identities.shape[0]) % 4 == 0

    X_train = embedded[train_idx]
    X_test = embedded[test_idx]

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


class Identity(object):

    __slots__ = ('name', 'path')

    def __init__(self, name, path):
        self.name = name  # person name
        self.path = path  # image path

    def __repr__(self):
        return f"Identity(name='{self.name}', path='{self.path}')"


def align_image(alignment, image):
    return alignment.align(
        96, image, alignment.get_largest_face_bounding_box(image),
        landmark_indices=AlignDlib.OUTER_EYES_AND_NOSE)


def load_alignment_lib(filename=ALIGNMENT_DATA_FILENAME):
    """Initialize the OpenFace face alignment utility."""
    if not os.path.exists(filename):
        url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        decompressor = bz2.BZ2Decompressor()
        with urlopen(url) as src, open(filename, 'wb') as dst:
            data = src.read(1024)
            while len(data) > 0:
                dst.write(decompressor.decompress(data))
                data = src.read(1024)
    return AlignDlib(filename)


def load_identities(path, patterns=('*.jpg', '*.jpeg'), **kwargs):
    identities = []
    for name in os.listdir(path):
        for filename in filesystem.find_recursive(os.path.join(path, name), patterns, **kwargs):
            identities.append(Identity(name, filename))
    return np.array(identities)


def load_image(path):
    """Reverse channels because OpenCV loads images in BGR mode."""
    return cv2.imread(path, 1)[..., ::-1]


def load_model():
    model = create_model()
    model.load_weights(os.path.join(face_reco_path, 'weights', 'nn4.small2.v1.h5'))
    return model


if __name__ == '__main__':
    main()
