# %matplotlib inline
# -*- coding: utf-8 -*-
import sys
import os
import os.path
from os import listdir
from PIL import Image
import numpy as np
from sklearn import metrics
from sklearn.decomposition import NMF, PCA
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
            labels.append(float(class_dict[image_name]))
        except KeyError:
            pass

    return imgs, labels


def NMF_decomposition(images):
    model = NMF(n_components=256, init='random', random_state=0)
    W = model.fit_transform(images)
    # H = model.components_
    print("shape after NMF: ", W[0].shape)

    return W


def PCA_decomposition(images):
    model = PCA(n_components=256)
    reduced_images = model.fit_transform(images)
    print("shape after PCA: ", reduced_images[0].shape)

    return reduced_images


def SVD_decomposition(train_images, test_images):
    transpose = np.transpose(train_images)
    [U, S, Vtranspose] = np.linalg.svd(transpose)
    SA = np.zeros((len(U), len(Vtranspose)), dtype=complex)
    SA[:len(S), :len(S)] = np.diag(S)
    component = 256
    SA_select = SA[0:component, 0:component]
    invSA_select = np.linalg.inv(SA_select)
    U_select = U[:, 0:component]
    V_select = Vtranspose[:, 0:component]
    train_images_reduced = V_select.tolist()

    result = []
    for img in test_images:
        tt = np.transpose(img)
        tmp = np.dot(np.dot(tt, U_select), invSA_select)
        result.append(tmp)

    rlist = []
    for r in result:
        rlist.append(r.tolist())

    test_images_reduced = []
    for ele in rlist:
        rctmp = []
        for e in ele:
            rctmp.append(e.real)
        test_images_reduced.append(rctmp)

    print("shape after SVD: ", train_images_reduced[0].shape)
    return train_images_reduced, test_images_reduced


if __name__ == "__main__":
    data_dict = {
        0: "training_sun_rose",
        1: "testing_sun_rose",
        2: "training_dan_tul",
        3: "testing_dan_tul"
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

    print("length of training data: ", len(training_data), len(training_label))
    print("example of training data: ", training_data[0])
    print("original shape: ", training_data[0].shape)

    print(">>> Decomposition reduction")
    decomposition_method = sys.argv[1]
    if decomposition_method == 'NMF':
        X_train_reduced = NMF_decomposition(training_data)
        X_test_reduced = NMF_decomposition(testing_data)
    elif decomposition_method == 'PCA':
        X_train_reduced = PCA_decomposition(training_data)
        X_test_reduced = PCA_decomposition(testing_data)
    else:  # SVD
        X_train_reduced, X_test_reduced = SVD_decomposition(training_data, testing_data)

    # classifier
    print(">>> Classifier")
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train_reduced, training_label)

    # classification result
    print(">>> Classification")
    score = clf.score(X_test_reduced, testing_label)
    print("score: ", score)
    predicted = clf.predict(X_test_reduced)
    print("confusion matrix:\n%s" % metrics.confusion_matrix(testing_label, predicted))
