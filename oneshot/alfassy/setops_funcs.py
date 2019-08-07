import numpy as np
import torch


def set_subtraction_operation(labels1, labels2):
    batch_size = labels1.shape[0]
    classesNum = labels1.shape[1]
    # print("labels1: ", labels1)
    # print("labels2: ", labels2)
    subLabels = []
    for vecNum in range(batch_size):
        subLabelPerClass = []
        for classNum in range(classesNum):
            if (labels1[vecNum][classNum] == 1) and (labels2[vecNum][classNum] == 0):
                subLabelPerClass += [1]
            else:
                subLabelPerClass += [0]
        subLabels += [subLabelPerClass]
    # print(subLabels)
    npSubLabels = np.asarray(subLabels)
    # print(npSubLabels)
    torSubLabels = torch.from_numpy(npSubLabels)
    # print(torSubLabels)
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


def set_subtraction_operation_one_sample(labels1, labels2):
    classesNum = labels1.shape[0]
    # print("labels1: ", labels1)
    # print("labels2: ", labels2)
    subLabelPerClass = []
    for classNum in range(classesNum):
        if (labels1[classNum] == 1) and (labels2[classNum] == 0):
            subLabelPerClass += [1]
        else:
            subLabelPerClass += [0]
    # print(subLabels)
    npSubLabels = np.asarray(subLabelPerClass)
    # print(npSubLabels)
    # subLabelPerClass = torch.from_numpy(subLabelPerClass)
    # print(torSubLabels)
    return npSubLabels


def set_union_operation_one_sample(labels1, labels2):
    classesNum = labels1.shape[0]
    subLabelPerClass = []
    for classNum in range(classesNum):
        if (labels1[classNum] == 1) or (labels2[classNum] == 1):
            subLabelPerClass += [1]
        else:
            subLabelPerClass += [0]
    npSubLabels = np.asarray(subLabelPerClass)
    # torSubLabels = torch.from_numpy(npSubLabels)
    return npSubLabels


def set_intersection_operation_one_sample(labels1, labels2):
    classesNum = labels1.shape[0]
    subLabelPerClass = []
    for classNum in range(classesNum):
        if (labels1[classNum] == 1) and (labels2[classNum] == 1):
            subLabelPerClass += [1]
        else:
            subLabelPerClass += [0]
    npSubLabels = np.asarray(subLabelPerClass)
    # torSubLabels = torch.from_numpy(npSubLabels)
    return npSubLabels