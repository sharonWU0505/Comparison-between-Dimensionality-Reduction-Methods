import sys
import os.path
import numpy as np
from os import listdir
from PIL import Image
from sklearn.decomposition import NMF
from sklearn import svm


def load_images(folder_path):
    img_paths = listdir(folder_path)
    pixels = []

    for img_path in img_paths[:1000]:
        path = os.path.join(folder_path, img_path)
        if not img_path.startswith('.') and os.path.isfile(path):
            img = Image.open(path).convert('RGB')
            arr = np.array(img).astype('float32')
            flat_arr = arr.ravel() / 255
            pixels.append(flat_arr)

    return np.array(pixels)


def load_labels(folder_path):
    img_paths = listdir(folder_path)
    labels = []

    for img_path in img_paths[:1000]:
        image_name = img_path[img_path.rfind("/") + 1: img_path.rfind("_")]
        labels.append(float(class_dict[image_name]))

    return labels


def NMF_decomposition(images):
    model = NMF(n_components=256, init='nndsvda', random_state=0)
    W = model.fit_transform(images)
    # H = model.components_

    return W


# def SVD_decomposition(images):
#
#
# def PCA_decomposition(images):


if __name__ == "__main__":
    dir = sys.argv[1]

    data_dict = {
        0: "training",
        1: "testing",
        2: "padding_training",
        3: "padding_testing"
    }

    class_dict = {
        "daisy": 0,
        "dandelion": 1,
        "rose": 2,
        "sunflower": 3,
        "tulip": 4
    }

    # load training data and do decomposition
    train_dir = dir + data_dict[2]
    train_labels = load_labels(train_dir)
    train_images = load_images(train_dir)
    train_data = NMF_decomposition(train_images)

    # classify
    classifier = svm.SVC(gamma=0.001)
    classifier.fit(train_data, train_labels)

    test_dir = dir + data_dict[3]
    test_data = NMF_decomposition(load_images(test_dir))
    predicted = classifier.predict(test_data)
    expected = load_labels(test_dir)

    print('predicted: ', predicted[:30])
    print('expected: ', expected[:30])
