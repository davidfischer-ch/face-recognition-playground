#!/usr/bin/env python3

# https://krasserm.github.io/2018/02/07/deep-face-recognition/
# http://nbviewer.jupyter.org/github/krasserm/face-recognition/blob/master/face-recognition.ipynb?flush_cache=true

import bz2, os
from urllib.request import urlopen

import cv2, numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

import face
from face.align import AlignDlib
from face.model import create_model

face_path = face.__path__._path[0]

ALIGNMENT_DATA_FILENAME = os.path.join(os.path.dirname(__file__), 'landmarks.dat')


def main():

    # Initialize
    alignment = load_alignment_lib()
    model = load_model()
    metadata = load_metadata(os.path.join(face_path, 'images'))
    embedded = np.zeros((metadata.shape[0], 128))
    for i, m in enumerate(metadata):
        image = load_image(m.image_path())
        image = align_image(alignment, image)
        # scale RGB values to interval [0,1]
        image = (image / 255).astype(np.float32)
        # obtain embedding vector for image
        embedded[i] = model.predict(np.expand_dims(image, axis=0))[0]

    # Train
    targets = np.array([m.name for m in metadata])
    encoder = LabelEncoder()
    encoder.fit(targets)

    # Numerical encoding of identities
    y = encoder.transform(targets)

    test_idx = np.arange(metadata.shape[0]) % 4 != 0
    train_idx = np.arange(metadata.shape[0]) % 4 == 0

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


class IdentityMetadata(object):

    __slots__ = ('base', 'name', 'file')

    def __init__(self, base, name, file):
        self.base = base  # dataset base directory
        self.name = name  # identity name
        self.file = file  # image file name

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


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


def load_image(path):
    """Reverse channels because OpenCV loads images in BGR mode."""
    return cv2.imread(path, 1)[..., ::-1]


def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)


def load_model():
    model = create_model()
    model.load_weights(os.path.join(face_path, 'weights', 'nn4.small2.v1.h5'))
    return model


if __name__ == '__main__':
    main()
