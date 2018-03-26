# %matplotlib inline
# -*- coding: utf-8 -*-
import sys
import os
import os.path
from os import listdir
from PIL import Image
#from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.decomposition import NMF
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

def load_data(folder_path, class_dict):
    img_paths = listdir(folder_path)
    imgs = []
    labels = []

    for img_path in img_paths:
        # get images
        path = os.path.join(folder_path, img_path)
        if not img_path.startswith('.') and os.path.isfile(path):
            img = Image.open(path).convert('LA')
            imgs.append(np.array(img).ravel())

        # get labels
        image_name = img_path[img_path.rfind("/") + 1: img_path.rfind("_")]
        try:
            labels.append(float(class_dict[image_name]))  # turn to float?
        except KeyError:
            pass

    return imgs, labels


def NMF_decomposition(images):
    model = NMF(n_components=256, init='random', random_state=0)
    W = model.fit_transform(images)
    # H = model.components_

    return W


if __name__ == "__main__":
    data_dict = {
        0: "training_sun_rose",
        1: "testing_sun_rose",
        2: "training_dan_tul",
        3: "testing_dan_tul",
        4: "p_training_sun_rose",
        5: "p_testing_sun_rose",
        6: "p_training_dan_tul",
        7: "p_testing_dan_tul"
    }

    class_dict_sun_rose = {
        "sunflower": 0,
        "rose": 1
    }

    class_dict_dan_tul = {
        "dandelion": 0,
        "tulip": 1
    }

    # load data
    print(">>> Load data")
    # sunflower and rose
    training_paths = os.getcwd() + '/data/' + data_dict[0]
    training_data, training_label = load_data(training_paths, class_dict_sun_rose)
    testing_paths = os.getcwd() + '/data/' + data_dict[1]
    testing_data, testing_label = load_data(testing_paths, class_dict_sun_rose)

    # # dandelion and tulip
    # training_paths = os.getcwd() + '/data/' + data_dict[2]
    # training_data, training_label = load_data(training_paths, class_dict_dan_tul)
    # testing_paths = os.getcwd() + '/data/' + data_dict[3]
    # testing_data, testing_label = load_data(testing_paths, class_dict_dan_tul)

    print(training_data[0])
    print("original shape: ", training_data[0].shape)
    print("length of training data: ", len(training_data), len(training_label))

    print(">>> Decomposition reduction")
    X_train_reduced = NMF_decomposition(training_data)
    print("shape after reduction: ", X_train_reduced[0].shape)

    # classification
    print(">>> Classification")
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train_reduced, training_label)

    print(">>> Decompose testing data")
    X_test_reduced = NMF_decomposition(testing_data)
    score = clf.score(X_test_reduced, testing_label)
    print("Score: ", score)

    predicted = clf.predict(X_test_reduced)
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(testing_label, predicted))
