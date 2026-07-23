from sklearn.metrics import (
    accuracy_score,
    classification_report as sklearn_classification_report,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def calculate_precision(y_true, y_pred, average="weighted", zero_division=0):
    return precision_score(y_true, y_pred, average=average, zero_division=zero_division)


def calculate_recall(y_true, y_pred, average="weighted", zero_division=0):
    return recall_score(y_true, y_pred, average=average, zero_division=zero_division)


def calculate_f1(y_true, y_pred, average="weighted", zero_division=0):
    return f1_score(y_true, y_pred, average=average, zero_division=zero_division)


def calculate_roc_auc(y_true, y_score, average="weighted", multi_class="ovr"):
    return roc_auc_score(y_true, y_score, average=average, multi_class=multi_class)


def classification_report(y_true, y_pred, target_names=None, output_dict=True, zero_division=0):
    return sklearn_classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=zero_division,
    )


def confusion_matrix(y_true, y_pred, labels=None):
    return sklearn_confusion_matrix(y_true, y_pred, labels=labels)
