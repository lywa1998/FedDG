import numpy as np 
import matplotlib.pyplot as plt 
import cv2
from sklearn.cluster import KMeans
k = 3;#number of dominating colors you want

def kmeans_cluster(path1, path2):
	img = cv2.imread(path1)#BGR format
	origin = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	truth = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

	width = img.shape[0]
	height = img.shape[1]

	img_f = img.reshape(-1,3)
	print(img_f.shape)

	classifier = KMeans(n_clusters=k)
	kmeans = classifier.fit(img_f)

	labels = kmeans.labels_

	prediction = labels.reshape((width, height))
	prediction[prediction == 0] = 0
	prediction[prediction == 1] = 125
	prediction[prediction == 2] = 255
	return origin, truth, prediction.astype(np.uint8)

def label_prediction(prediction, la):
	print(la)
	prediction_copy = prediction.copy()
	prediction[prediction_copy == 0] = la[0]
	prediction[prediction_copy == 125] = la[1]
	prediction[prediction_copy == 255] = la[2]
	return prediction

origin1, truth1, prediction1 = kmeans_cluster("G-1-L.png", "G-1-L_copy.png")
plt.imshow(prediction1, cmap="gray")
plt.show()
la = input("")
la = [int(n) for n in la.split()]
prediction1 = label_prediction(prediction1, la)


origin2, truth2, prediction2 = kmeans_cluster("g0003.png", "g0003_copy.png")
plt.imshow(prediction2, cmap="gray")
plt.show()
la = input("")
la = [int(n) for n in la.split()]
prediction2 = label_prediction(prediction2, la)


origin3, truth3, prediction3 = kmeans_cluster("gdrishtiGS_001.png", "gdrishtiGS_001_copy.png")
plt.imshow(prediction3, cmap="gray")
plt.show()
la = input("")
la = [int(n) for n in la.split()]
prediction3 = label_prediction(prediction3, la)


origin4, truth4, prediction4 = kmeans_cluster("V0001.png", "V0001_copy.png")
plt.imshow(prediction4, cmap="gray")
plt.show()
la = input("")
la = [int(n) for n in la.split()]
prediction4 = label_prediction(prediction4, la)

plt.subplot(431)
plt.imshow(origin1)
plt.xticks([]), plt.yticks([])
plt.title("Origin")
plt.subplot(432)
plt.imshow(truth1, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.title("Truth")
plt.subplot(433)
plt.imshow(prediction1, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.title("K mean")

plt.subplot(434)
plt.imshow(origin2)
plt.xticks([]), plt.yticks([])
plt.subplot(435)
plt.imshow(truth2, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.subplot(436)
plt.imshow(prediction2, cmap="gray")
plt.xticks([]), plt.yticks([])

plt.subplot(437)
plt.imshow(origin3)
plt.xticks([]), plt.yticks([])
plt.subplot(438)
plt.imshow(truth3, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.subplot(439)
plt.imshow(prediction3, cmap="gray")
plt.xticks([]), plt.yticks([])

plt.subplot(4,3, 10)
plt.imshow(origin4)
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 11)
plt.imshow(truth4, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 12)
plt.imshow(prediction4, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.show()