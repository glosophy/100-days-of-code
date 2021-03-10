"""
The F1 score is a commonly-used metric for evaluating the performance of classification models.
Given two vectors representing binary (0/1) labels with one representing the true labels and one representing the
predicted labels, compute the F1 score between them. Note, here "1" represents the positive label.
"""

VEC_TRUE = [0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
VEC_PREDICTED = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1]


def f1_score(true, predicted):
    """
    Compute the F1 score between two vectors of binary labels.
    :param true: Vector of true labels represented as list of integers
    :param predicted: Vector of predicted labels represented as list of integers
    :return: The F1 score between the vectors
    """

    TP = 0
    FP = 0
    FN = 0

    for i in range(len(true)):
        if true[i] == 1 and predicted[i] == 1:
            TP += 1
        if true[i] == 0 and predicted[i] == 1:
            FP += 1
        if true[i] == 1 and predicted[i] == 0:
            FN += 1

    F1 = TP / (TP + (0.5 * (FP + FN)))

    return F1


print(f1_score(VEC_TRUE, VEC_PREDICTED))
