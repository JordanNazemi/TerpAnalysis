import numpy as np
from sklearn import tree
import pandas as pd

train_data = pd.read_csv("Data/train_data.csv")
dev_data = pd.read_csv("Data/dev_data.csv")
test_data = pd.read_csv("Data/test_data.csv")

# visualizing the dataset
train_data

### checking for missing data
print(train_data.isnull().sum())
print(dev_data.isnull().sum())
print(test_data.isnull().sum())

# deleting blank columns
train_data.dropna(axis='columns', how='all', inplace=True)
dev_data.dropna(axis='columns', how='all', inplace=True)
test_data.dropna(axis='columns', how='all', inplace=True)
# deleting rows contained a blank feature
train_data.dropna(inplace=True)
dev_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# splitting the datasets into x and y vectors
dev_classes = dev_data.iloc[:, -3]  # just aroma
dev_features = dev_data.iloc[:, 2:-3]  # droping string data values at the beginning of the list

train_classes = train_data.iloc[:, -3]  # just aroma
train_features = train_data.iloc[:, 2:-3]  # droping string data values at the beginning of the list

test_classes = test_data.iloc[:, -3]  # just aroma
test_features = test_data.iloc[:, 2:-3]  # droping string data values at the beginning of the list

# used to turn aromas from a string to a list
def get_array_from_string(entry):
    complete_list = []
    entry = entry.replace("[", "")
    entry = entry.replace("]", "")
    entry = entry.replace("'", "")
    entry = entry.replace(" ", "")
    entry_array = entry.split(",")

    for aroma in entry_array:
        complete_list.append(aroma)

    return complete_list


# accepts a dataframe of string terpene classifications
# returns an array of containing 10 binary categorical aroma values
def terpine_one_hot_encoder_simpler(df):
    output = np.zeros((len(df.index), 10))
    i = 0
    for element in df:
        arr = get_array_from_string(element)
        for entry in arr:
            if entry == "Chemical" or entry == "Fuel" or entry == "Diesel":
                output[i][0] = 1
            if entry == "Earthy":
                output[i][1] = 1
            if entry == "Woody" or entry == "Pine":
                output[i][2] = 1
            if entry == "Citrus" or entry == "Lemon" or entry == "Lime" or entry == "Orange" or entry == "Citrus":
                output[i][3] = 1
            if entry == "Fruity" or entry == "Pineapple" or entry == "Strawberry" or entry == "Cherry" or entry == "Apple" or entry == "Grapefruit" or entry == "Mango" or entry == "Blueberry" or entry == "Grape" or entry == "Berry" or entry == "Tropical":
                output[i][4] = 1
            if entry == "Skunky" or entry == "Grassy" or entry == "Musky" or entry == "Dank" or entry == "Pungent" or entry == "Cheese" or entry == "Fragrant" or entry == "Kush" or entry == "Harsh":
                output[i][5] = 1
            if entry == "Nutty":
                output[i][6] = 1
            if entry == "Sweet" or entry == "Vanilla" or entry == "BubbleGum" or entry == "Candy" or entry == "Creamy" or entry == "Caramel" or entry == "Chocolate" or entry == "Mint":
                output[i][7] = 1
            if entry == "Spicy" or entry == "Pepper" or entry == "Hash":
                output[i][8] = 1
            if entry == "Herbal" or entry == "Floral" or entry == "Coffee" or entry == "Flowery" or entry == "Sage":
                output[i][9] = 1
        i += 1
    return output


def accuracy(predictions, actual):
    incorrect = 0
    incorrect_list = []
    partially_correct = 0
    partially_correct_list = []
    totally_correct = 0
    totally_correct_list = []
    correct = 0
    total_correct = 0
    total_aromas = 0
    for i in range(0, len(predictions)):
        for j in range(0, len(predictions[0])):
            # print("predictions[i][j]: " + str(predictions[i][j]))
            # print("actual[i][j]: " + str(actual[i][j]))
            if predictions[i][j] == actual[i][j] == 1:
                correct += 1
                total_correct += 1
                # print("correct: " + str(correct))
        if correct == np.count_nonzero(actual[i] == 1):
            totally_correct += 1
            totally_correct_list.append(i)
        elif correct > 0:
            partially_correct += 1
            partially_correct_list.append(i)
        else:
            incorrect += 1
            incorrect_list.append(i)

        total_aromas += np.count_nonzero(actual[i] == 1)
        correct = 0

    return total_correct / total_aromas


# one-hot-encoding aroma classes
train_classes_ohe = terpine_one_hot_encoder_simpler(train_classes)
dev_classes_ohe = terpine_one_hot_encoder_simpler(dev_classes)
test_classes_ohe = terpine_one_hot_encoder_simpler(test_classes)

# simpler-one-hot-encoding aroma classes
train_classes_ohe2 = terpine_one_hot_encoder_simpler(train_classes)
dev_classes_ohe2 = terpine_one_hot_encoder_simpler(dev_classes)
test_classes_ohe2 = terpine_one_hot_encoder_simpler(test_classes)

# grid search for simplified aromas using depth, minimum samples to split, and minimum samples per leaf
best_accuracy = 0
best_max_depth = 0
best_max_features = 0
for i in range(1, 25):
    for j in range(1, 30):
        clf = tree.DecisionTreeClassifier(max_depth=i, max_features=j, random_state=0)
        clf = clf.fit(train_features, train_classes_ohe2)
        predictions = clf.predict(dev_features)
        if (accuracy(predictions, dev_classes_ohe2)) > best_accuracy:
            best_accuracy = accuracy(predictions, dev_classes_ohe2)
            best_max_depth = i
            best_max_features = j
print(f"Best max depth: {best_max_depth}")
print(f"Best max depth: {best_max_features}")
print(f"Best accuracy: {best_accuracy}")

clf2 = tree.DecisionTreeClassifier(max_depth=17, max_features=22, random_state=0)
clf2 = clf2.fit(train_features, train_classes_ohe2)
predictions2 = clf2.predict(test_features)
accuracy(predictions2, test_classes_ohe2)
