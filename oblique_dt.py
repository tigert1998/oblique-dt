import warnings
import itertools

import numpy as np
from sklearn import svm


class ObliqueDT:
    splits_weight: np.array
    splits_bias: np.array
    labels: np.array

    @staticmethod
    def build_from_16_centroids(centroids: np.array):
        _, d = centroids.shape
        splits_weight = np.empty((15, d))
        splits_bias = np.empty((15,))
        labels = np.empty((16, ))

        indices = [None for _ in range(15)]
        indices[0] = list(range(16))
        for i in range(15):
            x = centroids[indices[i], :]
            l = len(indices[i])
            max_score = 0
            max_sep = 0
            for comb in itertools.combinations(range(l), l // 2):
                y = np.ones((l, ), dtype=np.int32)
                y[list(comb)] = 0
                clf = svm.SVC(kernel='linear', C=1e4, max_iter=20)
                clf.fit(x, y)
                score = np.mean(clf.predict(x) == y)
                sep = 1 / np.linalg.norm(clf.coef_, 2)

                if score > max_score or score == max_score and sep > max_sep:
                    max_score = score
                    max_sep = sep
                    splits_weight[i] = clf.coef_
                    splits_bias[i] = clf.intercept_
                    if i < 7:
                        indices[i * 2 + 1] = \
                            list(np.array(indices[i])[list(comb)])
                        indices[i * 2 + 2] = \
                            list(set(indices[i]).difference(
                                indices[i * 2 + 1]))
                    else:
                        labels[i * 2 + 1] = comb[0]
                        labels[i * 2 + 2] = \
                            list(set(indices[i]).difference(
                                indices[i * 2 + 1]))[0]
            if max_score < 1:
                warnings.warn("max_score < 1")

        dt = ObliqueDT()
        dt.splits_weight = splits_weight
        dt.splits_bias = splits_bias
        dt.labels = labels

        return dt
