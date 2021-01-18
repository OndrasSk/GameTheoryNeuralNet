from classifiers.probabilisticClassifier import *
from classifiers.notAllowHardFP import *
import sklearn.svm as svm

class ProbabilisticSoftSVM(ProbabilisticClassifier, NotAllowHardFP):

    def __init__(self, kernel, C=10, degree=3,**kwds):
        super().__init__(**kwds)
        self.c = C
        self.kernel = kernel
        self.degree = degree

    def classify_by_one_classifier(self, point, classifier):
        return classifier.predict_proba(point)[:,0]# .predict(point)

    def equals(self, c1, c2):
        return np.all(c1.support_vectors_ == c2.support_vectors_)

    def update(self, positive_data, positive_data_utility, att_support, game_matrix):
        data, validation_data, labels, weights = self.prepare_data(positive_data, positive_data_utility, att_support)
        data = data[weights>THETA]
        labels = labels[weights > THETA]
        weights = weights[weights>THETA]

        new_classifier = svm.SVC(kernel=self.kernel, degree=self.degree, C=self.c, probability=True)
        new_classifier.fit(data, labels, weights)
        return self.expand_classifiers(new_classifier, positive_data, positive_data_utility, att_support, game_matrix)