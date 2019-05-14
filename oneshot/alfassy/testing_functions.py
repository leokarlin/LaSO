import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


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


# ones only real vectors accuracy
def ones_only_real_vectors_accuracy(labels_x, labels_y, labels_z, ground_truth_x, ground_truth_y, ground_truth_z):
    batch_size = labels_x.shape[0]
    classes_num = labels_x.shape[1]
    correct_x = 0
    correct_y = 0
    correct_z = 0
    total = 0
    x_acc_list = []
    y_acc_list = []
    z_acc_list = []
    for i in range(batch_size):
        for j in range(classes_num):
            if (ground_truth_x[i][j] == 1) or (ground_truth_y[i][j] == 1):
                total += 1
                if labels_x[i][j] == ground_truth_x[i][j]:
                    correct_x += 1
                if labels_y[i][j] == ground_truth_y[i][j]:
                    correct_y += 1
                if labels_z[i][j] == ground_truth_z[i][j]:
                    correct_z += 1
        if total == 0:

            print("total == 0")
            print(ground_truth_x)
            print(ground_truth_y)
            print(i)
            print(total)
            print(correct_x)
            print(correct_y)
            print(correct_z)
            raise IndexError("in ones")
        x_acc_list += [correct_x / total]
        y_acc_list += [correct_y / total]
        z_acc_list += [correct_z / total]
        total = 0
        correct_x = 0
        correct_y = 0
        correct_z = 0
    x_mean = sum(x_acc_list)/len(x_acc_list)
    y_mean = sum(y_acc_list)/len(y_acc_list)
    z_mean = sum(z_acc_list)/len(z_acc_list)
    x_y_mean = (x_mean + y_mean)/2
    return x_y_mean, z_mean


# find the optimal intersection over union accuracy for different thresholds
def find_optimal_IOU_acc(data, valuesToSearch, ground_truth, is_fake):
    batch_size = data.shape[0]
    classes_num = data.shape[1]
    # print(valuesToSearch.shape)
    correct = 0
    total = 0
    item_acc_list = []
    th_acc_dict = {}
    for threshold in valuesToSearch:
        label_data = (data >= threshold)
        for i in range(batch_size):
            for j in range(classes_num):
                if (label_data[i][j] == 1) or (ground_truth[i][j] == 1):
                    total += 1
                    if label_data[i][j] == ground_truth[i][j]:
                        correct += 1
            if total == 0:
                if is_fake:
                    total = 0
                    correct = 0
                    item_acc_list += [1]
                    continue
                else:
                    print("in opt iou")
                    print("total == 0")
                    print(label_data)
                    print(ground_truth)
                    print(i)
                    print(correct)
                    print(total)
                    raise IndexError("real data")
            item_acc_list += [correct / total]
            total = 0
            correct = 0
        th_acc_dict[threshold] = sum(item_acc_list)/len(item_acc_list)
        item_acc_list = []
    maximum = max(th_acc_dict, key=th_acc_dict.get)  # Just use 'min' instead of 'max' for minimum.
    return maximum, th_acc_dict[maximum]


# check if this example is interesting in terms of object removal for the subtraction operator
def is_remove_interesting_sub(ground_truth, labels1):
    classes_num = ground_truth.shape[0]
    is_interesting = False
    interesting_indices = []
    for j in range(classes_num):
        if (labels1[j] == 1) and (ground_truth[j] == 0):
            is_interesting = True
            interesting_indices += [j]
    return is_interesting, interesting_indices


# check if this example is interesting in terms of object removal for the intersection operator
def is_remove_interesting_inter(ground_truth, labels1, labels2):
    classes_num = ground_truth.shape[0]
    is_interesting = False
    interesting_indices = []
    for j in range(classes_num):
        if ((labels1[j] == 1) or (labels2[j] == 1)) and (ground_truth[j] == 0):
            is_interesting = True
            interesting_indices += [j]
    return is_interesting, interesting_indices


# check if this example is interesting in terms of object removal for the subtraction/ intersection operator
def is_keep_interesting_sub(ground_truth, labels1):
    classes_num = ground_truth.shape[0]
    is_interesting = False
    interesting_indices = []
    for j in range(classes_num):
        if (labels1[j] == 1) and (ground_truth[j] == 1):
            is_interesting = True
            interesting_indices += [j]
    return is_interesting, interesting_indices


# check if this example is interesting in terms of object removal for the subtraction/ intersection operator
def is_keep_interesting(ground_truth, labels1):
    classes_num = ground_truth.shape[0]
    is_interesting = False
    interesting_indices = []
    for j in range(classes_num):
        if (labels1[j] == 1) and (ground_truth[j] == 1):
            is_interesting = True
            interesting_indices += [j]
    return is_interesting, interesting_indices


# did the generator remove the labels he was supposed to remove? for the subtraction operator
def labels_to_be_removed_acc_sub(data, ground_truth, labels1):
    classes_num = ground_truth.shape[0]
    correct = 0
    total = 0
    for j in range(classes_num):
        if (labels1[j] == 1) and (ground_truth[j] == 0):
            total += 1
            if data[j] == ground_truth[j]:
                correct += 1
    if total == 0:
        return 1
    item_acc = correct / total
    return item_acc


# did the generator remove the labels he was supposed to remove? for the intersection operator
def labels_to_be_removed_acc_inter(data, ground_truth, labels1, labels2):
    classes_num = ground_truth.shape[0]
    correct = 0
    total = 0
    for j in range(classes_num):
        if ((labels1[j] == 1) or (labels2[j] == 1)) and (ground_truth[j] == 0):
            total += 1
            if data[j] == ground_truth[j]:
                correct += 1
    if total == 0:
        return 1
    item_acc = correct / total
    return item_acc


def IOU_vectors_accuracy_1v1(ground_truth, result_vec):
    classes_num = ground_truth.shape[0]
    correct = 0
    total = 0

    for j in range(classes_num):
        if (result_vec[j] == 1) or (ground_truth[j] == 1):
            total += 1
            if result_vec[j] == ground_truth[j]:
                correct += 1
    if total == 0:
        return 1
    item_acc = correct / total
    return item_acc


def IOU_vectors_accuracy_1v1_person_nerf(ground_truth, result_vec):
    classes_num = ground_truth.shape[0]
    correct = 0
    total = 0

    for j in range(classes_num):
        if j == 7:
            continue
        if (result_vec[j] == 1) or (ground_truth[j] == 1):
            total += 1
            if result_vec[j] == ground_truth[j]:
                correct += 1
    if total == 0:
        return 1
    item_acc = correct / total
    return item_acc


def binary_per_sample_ret_accuracy(ground_truth, result_vec, keep_interesting, remove_interesting):
    classes_num = ground_truth.shape[0]
    item_acc = 1
    for j in range(classes_num):
        if (j in keep_interesting) or (j in remove_interesting):
            if result_vec[j] != ground_truth[j]:
                item_acc = 0
                break
    return item_acc


def precision_recall_statistics_backup(outputs_sig_np, targets):
    batch_size = outputs_sig_np.shape[0]
    classes_num = outputs_sig_np.shape[1]
    # print("classes num in precision recall: {}".format(classes_num))
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(classes_num):
        # precision[i], recall[i], _ = precision_recall_curve(targets[:, i], outputs_sig_np[:, i])
        # print("precision for class{}: {}, recall: {}".format(i, precision[i], recall[i]))
        # curve_name = "class" + str((i+1))
        # if should_plot:
        #     recall_precision_plotter.log(recall[i], precision[i], name=curve_name)
        average_precision[i] = average_precision_score(targets[:, i], outputs_sig_np[:, i], average="macro")

    # A "macro-average": quantifying score on all classes jointly
    # precision["macro"], recall["macro"], _ = precision_recall_curve(targets.ravel(), outputs_sig_np.ravel())
    average_precision["macro"] = average_precision_score(targets, outputs_sig_np, average="macro")
    # macro_average_plotter.log(recall['macro'], precision['macro'], name="macro_average")
    return average_precision


def precision_recall_statistics(outputs_sig_np, targets):
    batch_size = outputs_sig_np.shape[0]
    classes_num = outputs_sig_np.shape[1]
    # print("classes num in precision recall: {}".format(classes_num))
    # For each class
    # precision = dict()
    # recall = dict()
    average_precision = dict()
    average_precision_IOU = dict()
    for i in range(classes_num):
        # precision[i], recall[i], _ = precision_recall_curve(targets[:, i], outputs_sig_np[:, i])
        # print("precision for class{}: {}, recall: {}".format(i, precision[i], recall[i]))
        # curve_name = "class" + str((i+1))
        # if should_plot:
        #     recall_precision_plotter.log(recall[i], precision[i], name=curve_name)
        average_precision[i] = average_precision_score(targets[:, i], outputs_sig_np[:, i])
        filtered_tar, filtered_out = get_intersection(targets[:, i], outputs_sig_np[:, i])
        if filtered_tar.size == 0:
            average_precision_IOU[i] = -1
        else:
            # print("in precision average: ", filtered_tar)
            # print(filtered_tar.shape)
            # print(filtered_out.shape)
            average_precision_IOU[i] = average_precision_score(filtered_tar, filtered_out)
    # A "macro-average": quantifying score on all classes jointly
    # precision["macro"], recall["macro"], _ = precision_recall_curve(targets.ravel(), outputs_sig_np.ravel())
    average_precision["macro"] = average_precision_score(targets, outputs_sig_np, average="macro")
    # macro_average_plotter.log(recall['macro'], precision['macro'], name="macro_average")
    return average_precision, average_precision_IOU


def precision_recall_statistics_binary(outputs_sig_np, targets):
    batch_size = outputs_sig_np.shape[0]
    classes_num = outputs_sig_np.shape[1]
    # print("classes num in precision recall: {}".format(classes_num))
    # For each class
    # precision = dict()
    # recall = dict()
    # average_precision = dict()
    # average_precision_IOU = dict()
    # precision[i], recall[i], _ = precision_recall_curve(targets[:, i], outputs_sig_np[:, i])
    # print("precision for class{}: {}, recall: {}".format(i, precision[i], recall[i]))
    # curve_name = "class" + str((i+1))
    # if should_plot:
#     recall_precision_plotter.log(recall[i], precision[i], name=curve_name)
    average_precision = average_precision_score(targets, outputs_sig_np)
    # A "macro-average": quantifying score on all classes jointly
    # precision["macro"], recall["macro"], _ = precision_recall_curve(targets.ravel(), outputs_sig_np.ravel())
    # macro_average_plotter.log(recall['macro'], precision['macro'], name="macro_average")
    return average_precision


# Get smaller vectors which contain only the spots where at least one of the original vectors was 1
def get_intersection(targets, outputs):
    filtered_outputs = []
    filtered_targets = []
    for i in range(outputs.shape[0]):
        if (outputs[i] == 1) or (targets[i] == 1):
            filtered_outputs.append(outputs[i])
            filtered_targets.append(targets[i])
    return np.asarray(filtered_targets), np.asarray(filtered_outputs)


def is_intersection(labels1, labels2):
    res = False
    for label in labels1:
        if label in labels2:
            res = True
            break
    return res


def get_subtraction_exp(labels1, labels2):
    # res = [label for label in labels1 if label in labels2]
    res = []
    for label in labels1:
        if label not in labels2:
            res += [label]
    return res
