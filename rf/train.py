from tkinter.messagebox import NO
import cv2
import numpy as np
import pylab as plt
from glob import glob
import argparse
import os
import progressbar
import pickle as pkl
from numpy.lib import stride_tricks
from skimage import feature
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
import mahotas as mt

def check_args(args):

    if not os.path.exists(args.image_dir):
        raise ValueError("Image directory does not exist")

    if not os.path.exists(args.label_dir):
        raise ValueError("Label directory does not exist")

    if args.classifier != "SVM" and args.classifier != "RF" and args.classifier != "GBC":
        raise ValueError("Classifier must be either SVM, RF or GBC")

    if args.output_model.split('.')[-1] != "p":
        raise ValueError("Model extension must be .p")

    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir", default=r"pots/images/store" , help="Path to images")
    parser.add_argument("-l", "--label_dir", default=r"pots/labels/store", help="Path to labels")
    parser.add_argument("-c", "--classifier", default="RF", help="Classification model to use")
    parser.add_argument("-o", "--output_model", default="rf.p", help="Path to save model. Must end in .p")
    args = parser.parse_args()
    return check_args(args)

def read_data(image_dir, label_dir):

    print ('[INFO] Reading image data.')

    filelist = glob(os.path.join(image_dir, '*.jpg'))
    image_list = []
    label_list = []

    for file in filelist:

        image_list.append(cv2.imread(file, 1))
        label_list.append(cv2.imread(os.path.join(label_dir, os.path.basename(file).split('.')[0]+'.png'), 0))

    return image_list, label_list

def gabor_feature(img_gray):
    features = []
    num = 1 #To count numbers up in order to give Gabor features a lable in the data frame
    for theta in np.arange(0, np.pi, np.pi / 4): 
        for lamda in np.arange(-np.pi, np.pi, np.pi / 2): 
            gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
            ksize=9
            kernel = cv2.getGaborKernel((ksize, ksize), 2., theta, lamda, 0.5, 0, ktype=cv2.CV_32F)    
            fimg = cv2.filter2D(img_gray, cv2.CV_8UC3, kernel)
            filtered_img = fimg.reshape(-1, 1)
            features.append(filtered_img)
            print(gabor_label, ': theta=', theta, ': sigma=', 2, ': lamda=', lamda, ': gamma=', 0.5)
            num += 1  #Increment for gabor column label
    
    return np.hstack(features)

def edge_feature(img_gray):
    features = []
        #CANNY EDGE
    edges = cv2.Canny(img_gray, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1, 1)
    features.append(edges1)

    from skimage.filters import roberts, sobel, scharr, prewitt

    #ROBERTS EDGE
    edge_roberts = roberts(img_gray)
    edge_roberts1 = edge_roberts.reshape(-1, 1)
    features.append(edge_roberts1)

    #SOBEL
    edge_sobel = sobel(img_gray)
    edge_sobel1 = edge_sobel.reshape(-1, 1)
    features.append(edge_sobel1)

    #SCHARR
    edge_scharr = scharr(img_gray)
    edge_scharr1 = edge_scharr.reshape(-1, 1)
    features.append(edge_scharr1)

    #PREWITT
    edge_prewitt = prewitt(img_gray)
    edge_prewitt1 = edge_prewitt.reshape(-1, 1)
    features.append(edge_prewitt1)

    return np.hstack(features)

def filter_feature(img_gray):
    features = []
    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img_gray, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1, 1)
    features.append(gaussian_img1)

    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img_gray, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1, 1)
    features.append(gaussian_img3)

    #MEDIAN with sigma=3
    median_img = nd.median_filter(img_gray, size=3)
    median_img1 = median_img.reshape(-1, 1)
    features.append(median_img1)

    #VARIANCE with size=3
    variance_img = nd.generic_filter(img_gray, np.var, size=3)
    variance_img1 = variance_img.reshape(-1, 1)
    features.append(variance_img1)  #Add column to original dataframe

    return np.hstack(features)

def feature_extract(img, img_gray, label, train=True):

    features = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    gabor = gabor_feature(img_gray)
    edge = edge_feature(img_gray)
    filters = filter_feature(img_gray)
    features = np.hstack((features, gabor, edge, filters))

    if train == True:
        labels = label.reshape(label.shape[0]*label.shape[1], 1)
    else:
        labels = None

    return features, labels

def create_training_dataset(image_list, label_list):

    print ('[INFO] Creating training dataset on %d image(s).' %len(image_list))

    X = []
    y = []

    for i, img in enumerate(image_list):
        img = cv2.resize(img, (384, 384))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features, labels = feature_extract(img, img_gray, label_list[i])
        X.append(features)
        y.append(labels)

    X = np.array(X)
    X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
    y = np.array(y)
    y = y.reshape(y.shape[0]*y.shape[1], y.shape[2]).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print ('[INFO] Feature vector size:', X_train.shape)
    print ('[INFO] Label vector size:', y_train.shape)

    return X_train, X_test, y_train, y_test

def train_model(X, y, classifier):

    if classifier == "SVM":
        from sklearn.svm import SVC
        print ('[INFO] Training Support Vector Machine model.')
        model = SVC()
        model.fit(X, y)
    elif classifier == "RF":
        from sklearn.ensemble import RandomForestClassifier
        print ('[INFO] Training Random Forest model.')
        model = RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42)
        model.fit(X, y)
    elif classifier == "GBC":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        model.fit(X, y)

    print ('[INFO] Model training complete.')
    print ('[INFO] Training Accuracy: %.2f' %model.score(X, y))
    return model

def test_model(X, y, model):

    pred = model.predict(X)
    precision = metrics.precision_score(y, pred, average='weighted', labels=np.unique(pred))
    recall = metrics.recall_score(y, pred, average='weighted', labels=np.unique(pred))
    f1 = metrics.f1_score(y, pred, average='weighted', labels=np.unique(pred))
    accuracy = metrics.accuracy_score(y, pred)

    print ('--------------------------------')
    print ('[RESULTS] Accuracy: %.2f' %accuracy)
    print ('[RESULTS] Precision: %.2f' %precision)
    print ('[RESULTS] Recall: %.2f' %recall)
    print ('[RESULTS] F1: %.2f' %f1)
    print ('--------------------------------')

def main(image_dir, label_dir, classifier, output_model):

    start = time.time()

    image_list, label_list = read_data(image_dir, label_dir)
    X_train, X_test, y_train, y_test = create_training_dataset(image_list, label_list)
    model = train_model(X_train, y_train, classifier)
    print(model.feature_importances_)
    test_model(X_test, y_test, model)
    pkl.dump(model, open(output_model, "wb"))
    print ('Processing time:',time.time()-start)

if __name__ == "__main__":
    args = parse_args()
    image_dir = args.image_dir
    label_dir = args.label_dir
    classifier = args.classifier
    output_model = args.output_model
    main(image_dir, label_dir, classifier, output_model)
