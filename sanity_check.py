# these are all probabilities for the seed one of the MNIST dataset
import math
import re


def format_helper(prediction, target):
    probabilities = dict.fromkeys(["X{}Y{}".format(x,y) for x in set(prediction) for y in set(target)], 0)
    for i in range(len(prediction)):
        probabilities["X{}Y{}".format(prediction[i],target[i])] += 1
    print(probabilities)
    return probabilities


def get_joint_MI(probabilities=None, target=None, prediction=None):
    reg_X = re.compile(r'X[0-9]{1,}')
    reg_Y = re.compile(r'Y[0-9]{1,}')

    # This function checks that the given dictionary is formatted correctly
    # or if there are two, formats them
    if prediction is None and target is None:
        print("in formatted probabilities")
    elif probabilities is None:
        print("in unformatted")
        probabilities = format_helper(prediction, target)
    else:
        raise Exception("bad input")

    #these are a bit ugly, essentially they're searching each variable and creating a dictionary of unique keys X and Y
    x_var = dict.fromkeys(set([(re.match(reg_X, k)).group(0) for k, v in probabilities.items()]), 0)
    y_var = dict.fromkeys(set([(re.search(reg_Y, k)).group(0) for k, v in probabilities.items()]), 0)
    mi_sum = 0
    size = sum([v for k,v in probabilities.items()])

    #populating dictionaries with numbers
    for k, v in probabilities.items():
        temp_x = re.match(reg_X, k).group(0)
        temp_y = re.search(reg_Y, k).group(0)
        x_var[temp_x] += v / size
        y_var[temp_y] += v / size
        probabilities[k] = v / size

    for y_key, y_value in y_var.items():
        for x_key, x_value in x_var.items():
            prob_joint = probabilities.get("{}{}".format(x_key, y_key))
            mi_sum += prob_joint * math.log(prob_joint/(x_value*y_value))

    print(""
          "X", x_var,
          "\nY", y_var,
          "\nProbabilities", probabilities,
          "\nSum", mi_sum)

    return mi_sum

probability_dict = {
    'X11Y1': 351,
    'X10Y1': 24,
    'X01Y1': 357,
    'X00Y1': 160,
    'X11Y0': 5,
    'X10Y0': 121,
    'X01Y0': 24,
    'X00Y0': 8958
}

target = [0,1,1,0,0,0,1]
prediction = [0,1,0,1,0,0,1]

print("Single Dict ------------------------")
get_joint_MI(probabilities=probability_dict)
print("------------------------------------"
      ""
      "Target/Prediction-------------------")
get_joint_MI(target=target, prediction=prediction)
