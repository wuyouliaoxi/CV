import cv2
import numpy as np
import pickle
import os
import random
from scipy import spatial
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
import matplotlib.pyplot as plt


# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        # dimension of the embeddings is 128
        self.dim_embeddings = 128
        # Reads a network model ONNX using a ResNet-50 neural network architecture, return network object
        self.facenet = cv2.dnn.readNetFromONNX("resnet50_128.onnx")  # ResNet-50

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)  # cv2 is BGR
        # face = face - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.reshape(reshaped, (1, 3, 224, 224))

        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)  # feature normalization?

    # Get dimensionality of the extracted embeddings.
    def get_embedding_dimensionality(self):
        return self.dim_embeddings


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=11, max_distance=1.0, min_prob=0.5):
        # ToDo: Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob

        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Load face recognizer from pickle file if available.
        if os.path.exists("recognition_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("recognition_gallery.pkl", 'wb') as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("recognition_gallery.pkl", 'rb') as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # ToDo
    # takes a new aligned face of known class from a video,
    # extracts its embedding, and stores it as a training sample in the gallery.
    def update(self, face, label):
        embedding = self.facenet.predict(face)
        self.embeddings = np.append(self.embeddings, [embedding], axis=0)
        self.labels.append(label)

    # ToDo
    # assigns a class label to an aligned face using k-NN.
    def predict(self, face):
        # Closed-Set Protocol
        embedding = self.facenet.predict(face)
        embedding = embedding.reshape(1, 128)
        X = self.embeddings
        y = np.array(self.labels)

        neigh = KNeighborsClassifier(n_neighbors=self.num_neighbours + 1, algorithm='brute').fit(X, y)
        predicted_label = neigh.predict(embedding)
        prob = max(np.squeeze(neigh.predict_proba(embedding)))
        neigh_dist, _ = neigh.kneighbors(embedding)
        dist_to_prediction = min(np.squeeze(neigh_dist))

        # Open-Set Protocol
        if dist_to_prediction > self.max_distance or prob < self.min_prob:
            predicted_label = "unknown"

        return predicted_label, prob, dist_to_prediction


# The FaceClustering class enables unsupervised clustering of face images according to their identity and
# re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self, num_clusters=5, max_iter=25):
        # ToDo: Prepare FaceNet.
        self.facenet = FaceNet()
        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, self.facenet.get_embedding_dimensionality()))
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists("clustering_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("clustering_gallery.pkl", 'wb') as f:
            pickle.dump((self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("clustering_gallery.pkl", 'rb') as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = pickle.load(f)

    # ToDo
    # extracts and stores an embedding for a new face
    def update(self, face):
        embedding = self.facenet.predict(face)
        self.embeddings = np.append(self.embeddings, [embedding], axis=0)

    # ToDo
    # Implement the k-means algorithm
    # Store the estimated cluster centers and the labels assigned to the faces.
    def fit(self):

        # step1: initialization, choose k embeddings of the input data at random
        # shape：(k,128)
        self.cluster_center = self.embeddings[random.sample(range(len(self.embeddings)), self.num_clusters)]
        sum_dis = []
        # repeat until convergence or reach the maximum number of iterations
        for i in range(self.max_iter):
            # step2: for each point, find nearest centroid (distance calculation) and assign the point to this cluster
            self.cluster_membership = []
            for i in range(len(self.embeddings)):
                repeat_embedding = np.repeat([self.embeddings[i]], repeats=self.num_clusters, axis=0)  # shape：(5,128)
                # measure the euclidean distance between each point and each centroid
                distance = np.linalg.norm(repeat_embedding - self.cluster_center, axis=1)
                # Assign the each point to the nearest cluster
                center_idx = np.argmin(distance)
                # Cluster index associated with the different samples and store it in cluster_membership
                self.cluster_membership.append(center_idx)

            # step3: for each cluster, new centroid is mean of all points assigned to cluster in previous step
            intra_dis = []
            for j in range(self.num_clusters):
                idx = np.array(self.cluster_membership) == j
                # Convergence analysis
                # objective function: minimizes aggregate intra-cluster distance
                intra_dis.append(np.linalg.norm(self.cluster_center[j] - self.embeddings[idx]))
                self.cluster_center[j] = np.mean(self.embeddings[idx], axis=0)
            sum_dis.append(sum(intra_dis))
            print(sum_dis)

        # draw diagram
        x = np.arange(0, 25, 1)
        y = np.array(sum_dis)
        plt.plot(x, y)
        plt.xlabel('iteration')
        plt.ylabel('value of objective function')
        plt.title('convergence analysis')
        plt.show()

    # ToDo
    def predict(self, face):
        embedding = self.facenet.predict(face)
        # extend embedding to the same shape of cluster_center (5,128)
        repeat_embedding = np.repeat([embedding], repeats=self.num_clusters, axis=0)  # shape：(5,128)
        # The best matching cluster j has the smallest distance between its center and a given face
        distances_to_clusters = np.linalg.norm(self.cluster_center - repeat_embedding, axis=1)
        predicted_label_idx = np.argmin(distances_to_clusters)
        return predicted_label_idx, distances_to_clusters
