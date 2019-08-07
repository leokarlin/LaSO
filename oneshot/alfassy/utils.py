import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import torch
import logging
import shutil


# intersection over union real vectors accuracy
def IOU_real_vectors_accuracy(data, ground_truth):

    batch_size = data.shape[0]
    classes_num = data.shape[1]
    correct = 0
    total = 0
    item_acc_list = []
    for i in range(batch_size):
        ground_empty = True
        for j in range(classes_num):
            if (data[i][j] == 1) or (ground_truth[i][j] == 1):
                total += 1
                if ground_truth[i][j] == 1:
                    ground_empty = False
                if data[i][j] == ground_truth[i][j]:
                    correct += 1

        if ground_empty:
            print("in IOU real")
            print("ground_truth empty ")
            print(data)
            print(ground_truth)
            print(i)
            print(correct)
            print(total)
            raise IndexError("ground_truth is empty")
        if total == 0:
            print("in IOU real")
            print(data)
            print(ground_truth)
            print(i)
            print(j)
            print(correct)
            print(total)
        item_acc_list += [correct / total]
        total = 0
        correct = 0
    res = sum(item_acc_list)/len(item_acc_list)
    return res


# intersection over union fake vectors accuracy
def IOU_fake_vectors_accuracy(result_vec, ground_truth):
    batch_size = ground_truth.shape[0]
    classes_num = ground_truth.shape[1]
    correct = 0
    total = 0
    item_acc_list = []
    for i in range(batch_size):
        for j in range(classes_num):
            if (result_vec[i][j] == 1) or (ground_truth[i][j] == 1):
                total += 1
                if result_vec[i][j] == ground_truth[i][j]:
                    correct += 1
        if total == 0:
            # print("in iou fake")
            # print("total == 0")
            # print(result_vec)
            # print(ground_truth)
            # print(i)
            # print(correct)
            # print(total)
            # total = 0
            # correct = 0
            item_acc_list += [1]
            continue
        item_acc_list += [correct / total]
        total = 0
        correct = 0
    res = sum(item_acc_list)/len(item_acc_list)
    return res


def precision_recall_statistics(outputs_scores_np, targets):
    '''
    :param outputs_scores_np: numpy array with outputs scores
    :param targets: numpy array with targets
    :return: dictionary, average precision from precision recall graph for each class and macro averaged
    '''
    classes_num = outputs_scores_np.shape[1]
    average_precision = dict()
    for i in range(classes_num):
        average_precision[i] = average_precision_score(targets[:, i], outputs_scores_np[:, i])
    average_precision["macro"] = average_precision_score(targets, outputs_scores_np, average="macro")
    return average_precision


def get_subtraction_exp(labels1, labels2):
    # res = [label for label in labels1 if label in labels2]
    res = []
    for label in labels1:
        if label not in labels2:
            res += [label]
    return res


def set_subtraction_operation(labels1, labels2):
    batch_size = labels1.shape[0]
    classesNum = labels1.shape[1]

    subLabels = []
    for vecNum in range(batch_size):
        subLabelPerClass = []
        for classNum in range(classesNum):
            if (labels1[vecNum][classNum] == 1) and (labels2[vecNum][classNum] == 0):
                subLabelPerClass += [1]
            else:
                subLabelPerClass += [0]
        subLabels += [subLabelPerClass]
    npSubLabels = np.asarray(subLabels)
    torSubLabels = torch.from_numpy(npSubLabels)
    return torSubLabels


def set_union_operation(labels1, labels2):
    batch_size = labels1.shape[0]
    classesNum = labels1.shape[1]
    subLabels = []
    for vecNum in range(batch_size):
        subLabelPerClass = []
        for classNum in range(classesNum):
            if (labels1[vecNum][classNum] == 1) or (labels2[vecNum][classNum] == 1):
                subLabelPerClass += [1]
            else:
                subLabelPerClass += [0]
        subLabels += [subLabelPerClass]
    npSubLabels = np.asarray(subLabels)
    torSubLabels = torch.from_numpy(npSubLabels)
    return torSubLabels


def set_intersection_operation(labels1, labels2):
    batch_size = labels1.shape[0]
    classesNum = labels1.shape[1]
    subLabels = []
    for vecNum in range(batch_size):
        subLabelPerClass = []
        for classNum in range(classesNum):
            if (labels1[vecNum][classNum] == 1) and (labels2[vecNum][classNum] == 1):
                subLabelPerClass += [1]
            else:
                subLabelPerClass += [0]
        subLabels += [subLabelPerClass]
    npSubLabels = np.asarray(subLabels)
    torSubLabels = torch.from_numpy(npSubLabels)
    return torSubLabels


def configure_logging(log_filename):
    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.DEBUG)
    # Format for our log lines
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Setup console logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Setup file logging
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def save_checkpoint(state, is_best, epoch, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filePath = checkpoint + filename + 'epoch:' + str(epoch)
    torch.save(state, filePath)
    if is_best:
        best_filePath = checkpoint + filename + 'best'
        shutil.copyfile(filePath, best_filePath)


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr
