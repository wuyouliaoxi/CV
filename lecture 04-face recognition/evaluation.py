import numpy as np
import pickle
from classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(self,
                 classifier=NearestNeighborClassifier(),
                 false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True)):

        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

        self.similarity_thresholds = []
        self.similarity = []
        self.similarity_known = []

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):

        with open(train_data_file, 'rb') as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding='bytes')
        with open(test_data_file, 'rb') as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding='bytes')

    # Run the evaluation and find performance measure (identification rates) at different similarity thresholds.
    def run(self):
        # 1. Fit the classifier on the training data
        self.classifier.fit(self.train_embeddings, self.train_labels)
        # 2. Predict similarities on the test data.
        # prediction_labels, similarity, (10248,)
        # test_labels (10248,)
        # test_labels_know (4865,)
        # prediction_label_know (10248,)
        # prediction labels haven't unknown, that means false alarm (depending on threshold)
        prediction_labels, self.similarity = \
            self.classifier.predict_labels_and_similarities(self.test_embeddings)

        # delete unknown
        self.similarity_known = self.similarity[self.test_labels != UNKNOWN_LABEL]  # similarity of known label, (4865,)

        # 3. find a similarity threshold and compute the corresponding identification rate.
        # threshold decreases with FAR
        self.similarity_thresholds = self.select_similarity_threshold(self.similarity, self.false_alarm_rate_range)
        identification_rates = self.calc_identification_rate(prediction_labels)

        # Report all performance measures.
        evaluation_results = {'similarity_thresholds': self.similarity_thresholds,
                              'identification_rates': identification_rates}

        return evaluation_results

    # false alarm rate means erroneously recognized unknown
    # number of > threshold / number of unknown = FAR (FAR related to the unknown set)
    # open-set identification on the test data yields a given false alarm rate.
    def select_similarity_threshold(self, similarity, false_alarm_rate):
        return np.percentile(similarity[self.test_labels == UNKNOWN_LABEL], (1 - false_alarm_rate) * 100)

    def calc_identification_rate(self, prediction_labels):
        # here the input is actually prediction_labels_known
        prediction_labels_known = prediction_labels[self.test_labels != UNKNOWN_LABEL]
        identification_rate = []
        # test_labels_known = self.test_labels[self.test_labels != UNKNOWN_LABEL]
        # test_labels_unknown = self.test_labels[self.test_labels == UNKNOWN_LABEL]
        for t in self.similarity_thresholds:  # (1000,) also represents the x-axis (FAR)
            n_true = 0
            # detection (exceeding some threshold)
            for s in self.similarity_known:
                if s > t:
                    n_true += 1
            # identification, what means rank???
            # for prediction_label, test_label in zip(prediction_labels_known, test_labels_known):
            #     if prediction_label == test_label:
            #         n_true += 1
            dir = n_true / len(prediction_labels_known)  # DIR related to the known set
            identification_rate.append(dir)
        return identification_rate
