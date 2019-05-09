import re
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score, accuracy_score, precision_score, f1_score
import torch
import numpy as np
from collections import Counter
from scipy.stats import entropy
import random
from functools import reduce


def get_conditional_probability(prob_dict):
    """
    gets conditional probability for a dictionary of variables and outcomes

    :param: prob_dict
    :return: prob_dict
    """
    total = reduce(lambda x, y: x + y, prob_dict.values())
    prob_dict = {k: v / total for k, v in prob_dict.items()}
    print(prob_dict)
    return prob_dict


def get_prob_and_joint(prob_dict, prediction_CNN, prediction_Regression, target, size):
    """
    This horrifyingly ugly function creates a tensor of the joint output and populates a probability dictionary

    todo: fixing this function
    :param prob_dict:
    :param prediction_CNN:
    :param prediction_Regression:
    :param target:
    :param size:
    :return:
    """
    joint_output = torch.zeros(size)
    for i in range(len(target)):
        if prediction_CNN[i].item() == 1:
            if prediction_Regression[i].item() == 1:
                joint_output[i] = 0b11
                if target[i] == 1:
                    prob_dict['X11Y1'] = prob_dict.get('X11Y1', 0) + 1
                else:
                    prob_dict['X11Y0'] = prob_dict.get('X11Y0', 0) + 1
            else:
                joint_output[i] = 0b10
                if target[i] == 1:
                    prob_dict['X10Y1'] = prob_dict.get('X10Y1', 0) + 1
                else:
                    prob_dict['X10Y0'] = prob_dict.get('X10Y0', 0) + 1
        else:
            if prediction_Regression[i].item() == 1:
                joint_output[i] = 0b01
                if target[i] == 1:
                    prob_dict['X01Y1'] = prob_dict.get('X01Y1', 0) + 1
                else:
                    prob_dict['X01Y0'] = prob_dict.get('X01Y0', 0) + 1
            else:
                joint_output[i] = 0b00
                if target[i] == 1:
                    prob_dict['X00Y1'] = prob_dict.get('X00Y1', 0) + 1
                else:
                    prob_dict['X00Y0'] = prob_dict.get('X00Y0', 0) + 1
    return joint_output, prob_dict


def binary_target_helper(arr, target=5):
    """
    helper function for deciding between a binary target or an array target

    this function used to set an array to a binary array but now that testing requires more classes, it
    essentially exists because I was too worried (and lazy) to actually go and re-name everything

    so it uses the legacy version for integer targets, and the new one for lists of targets
    :param arr: the array to be changed
    :param target: the classes to change
    :return:
    """
    if type(target).__name__ == 'int':
        return binary_target(arr, target)
    else:
        return array_target(arr, target)


def binary_target(arr, target=5):
    """
    Legacy target helper

    changed an array (usually a target or an output) into only a few specific things

    if you wanted to look for fives, it would return a list or tensor where each instance of a five was instead a one
    """
    if type(arr).__name__ == 'Tensor':
        return torch.tensor([1 if target is arr[i].item() else 0 for i in range(len(arr))])
    else:
        return [1 if target is arr[i] else 0 for i in range(len(arr))]


def array_target(arr, target=[5, 6, 7]):
    """
    Target helper

    Like the legacy helper but for multiple classes
    """
    target_set = set(target)
    if type(arr).__name__ == 'Tensor':
        return torch.tensor([target.index(x) + 1 if x.item() in target_set else 0 for x in arr])
    else:
        return [target.index(x) + 1 if x in target_set else 0 for x in arr]


def entropy_helper(array):
    """
    Takes in an output tensor and returns the entropy

    The first two lines convert a tensor to a set of counts for each variable

    the last line formats that into a list that can be used by SKLearn's entropy function

    Process:
    1. clones an array and converts to numpy
    2. returns the counts
    :param array:
    :return:
    """
    counts = array.clone().cpu().numpy()
    counts = list(Counter(counts).values())
    return entropy(list(map(lambda x: x / len(array), counts)))


def array_checker(prediction_CNN_binary):
    counts = dict()
    for i in prediction_CNN_binary:
        counts[str(i.item())] = counts.get(str(i.item()), 0) + 1
    print(counts)


def calculate_mutual_info(model_CNN, model_Regression, test_loader, device, size, classes, single_batch=True):
    """
    This function calculates and returns a lot of information relating to the Mutual information of two models


    :param model_CNN: the CNN
    :param model_Regression: the Regression
    :param test_loader: the test data
    :param device: the device (almost always cuda)
    :param size: the size of the batch
    :param single_batch: deprecated
    :param classes:

    :return: mi_prediction_CNN_target, mi_prediction_REG_target: mutual information between the target and the CNN and REG models (respectively)
    :return: mi_joint_pred_target: mutual information between the joint output and the target
    :return: mi_redundancy: mutual information between the models
    :return: ent_joint: entropy of the joint output
    :return: ent_model_CNN, ent_model_REG: mutual information between the target and the CNN and REG models (respectively)
    :return: ent_target: entropy of the target
    :return: acc_CNN, acc_REG: Accuracy for the CNN and REG models (respectively)
    :return: prescision_CNN, prescision_REG: prescision for the CNN and REG models (respectively)
    :return: f1_CNN, f1_REG: f1 scores for the CNN and REG models (respectively)
    """
    # joint_output = np.zeros(size)
    prob_dict = {
        "X11Y1": 0,
        "X10Y1": 0,
        "X01Y1": 0,
        "X00Y1": 0,
        "X11Y0": 0,
        "X10Y0": 0,
        "X01Y0": 0,
        "X00Y0": 0
    }
    with torch.no_grad():
        target_dict = {}
        for index, (data, target) in enumerate(test_loader):
            # send to cuda
            data, target = data.to(device), target.to(device)
            # Predictions for the CNN and Logistic regression respectively
            # image is reshaped for the regression model because that makes it happy
            prediction_CNN = model_CNN(data).max(1, keepdim=True)[1]
            prediction_Regression = model_Regression(data.reshape(-1, 28 * 28)).max(1, keepdim=True)[1]

            # Binary conversion for target and CNN
            prediction_CNN_binary = binary_target_helper(prediction_CNN.clone(), classes)
            binary_target = binary_target_helper(target.clone(), classes)

            # Binary formatting for joint model as well as populating the probability dictionary
            # (very similar processes so They're consolidated
            joint_output, prob_dict = get_prob_and_joint(prob_dict, prediction_CNN_binary,
                                                         prediction_Regression, binary_target, size)
            # Mutual information scores
            mi_redundancy = mutual_info_score(prediction_CNN_binary.reshape(-1), prediction_Regression.reshape(-1))
            mi_prediction_CNN_target = mutual_info_score(binary_target, prediction_CNN_binary.reshape(-1))
            mi_prediction_REG_target = mutual_info_score(binary_target, prediction_Regression.reshape(-1))
            mi_joint_pred_target = mutual_info_score(binary_target, joint_output.reshape(-1))
            # array_sanity_check(prediction_CNN_binary, )
            # Entropy scores
            ent_joint = entropy_helper(joint_output)
            ent_model_CNN = entropy_helper(prediction_CNN_binary.reshape(-1))
            ent_model_REG = entropy_helper(prediction_Regression.reshape(-1))
            ent_target = entropy_helper(binary_target)

            # Accuracy scores
            acc_CNN = accuracy_score(binary_target, prediction_CNN_binary.reshape(-1))
            acc_REG = accuracy_score(binary_target, prediction_Regression.reshape(-1))

            # Prescision scores
            array_checker(prediction_CNN_binary)
            array_checker(prediction_Regression)
            # prescision_CNN = precision_score(binary_target, prediction_CNN_binary.reshape(-1))
            # prescision_REG = precision_score(binary_target, prediction_Regression.reshape(-1))
            # f1_CNN = f1_score(binary_target, prediction_CNN_binary.reshape(-1))
            # f1_REG = f1_score(binary_target, prediction_Regression.reshape(-1))
            prescision_CNN = 0
            prescision_REG = 0
            f1_CNN = 0
            f1_REG = 0

    # get_conditional_probability(prob_dict)
    # get_joint_MI(prob_dict)
    return mi_prediction_CNN_target.item(), \
           mi_prediction_REG_target.item(), \
           mi_joint_pred_target.item(), \
           mi_redundancy.item(), \
           ent_joint.item(), \
           ent_model_CNN.item(), \
           ent_model_REG.item(), \
           ent_target, \
           acc_CNN, acc_REG, \
           prescision_CNN, prescision_REG, \
           f1_CNN, f1_REG
