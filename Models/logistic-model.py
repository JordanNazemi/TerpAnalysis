import pandas as pd
import numpy as np
import scipy as sp
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import preprocessing
from random import seed
from random import random


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


def replace_aromas(df):
    for index, row in df.iterrows():
        arr = get_array_from_string(row["Aromas"])
        final_arr = []
        for entry in arr:
            if entry == "Chemical" or entry == "Fuel" or entry == "Diesel":
                if "diesel" not in final_arr:
                    final_arr.append("diesel")
            if entry == "Earthy":
                if "earthy" not in final_arr:
                    final_arr.append("earthy")
            if entry == "Woody" or entry == "Pine":
                if "pine" not in final_arr:
                    final_arr.append("pine")
            if entry == "Citrus" or entry == "Lemon" or entry == "Lime" or entry == "Orange" or entry == "Citrus":
                if "citrus" not in final_arr:
                    final_arr.append("citrus")
            if entry == "Fruity" or entry == "Pineapple" or entry == "Strawberry" or entry == "Cherry" or entry == "Apple" or entry == "Grapefruit" or entry == "Mango" or entry == "Blueberry" or entry == "Grape" or entry == "Berry" or entry == "Tropical":
                if "fruity" not in final_arr:
                    final_arr.append("fruity")
            if entry == "Skunky" or entry == "Grassy" or entry == "Musky" or entry == "Dank" or entry == "Pungent" or entry == "Cheese" or entry == "Fragrant" or entry == "Kush" or entry == "Harsh":
                if "skunky" not in final_arr:
                    final_arr.append("skunky")
            if entry == "Nutty":
                if "nutty" not in final_arr:
                    final_arr.append("nutty")
            if entry == "Sweet" or entry == "Vanilla" or entry == "BubbleGum" or entry == "Candy" or entry == "Creamy" or entry == "Caramel" or entry == "Chocolate" or entry == "Mint":
                if "sweet" not in final_arr:
                    final_arr.append("sweet")
            if entry == "Spicy" or entry == "Pepper" or entry == "Hash":
                if "spicy" not in final_arr:
                    final_arr.append("spicy")
            if entry == "Herbal" or entry == "Floral" or entry == "Coffee" or entry == "Flowery" or entry == "Sage":
                if "herbal" not in final_arr:
                    final_arr.append("herbal")
        df.at[index, "Aromas"] = final_arr

    return df


def get_aroma_dict(df):
    aroma_list = []
    complete_list = []
    for entry in df:
        for aroma in entry:
            if aroma not in aroma_list and aroma != "":
                aroma_list.append(aroma)
            complete_list.append(aroma)

    aroma_dict = {}
    for aroma in aroma_list:
        for entry in complete_list:
            if entry == aroma:
                if aroma in aroma_dict:
                    aroma_dict[aroma] = aroma_dict[aroma] + 1
                else:
                    aroma_dict[aroma] = 1

    return aroma_dict


def train_all_vs_one(train_features, train_classes, dev_features, dev_classes):
    dict = get_aroma_dict(train_classes)
    train_classes = train_classes.to_frame()
    dev_classes = dev_classes.to_frame()
    model_dict = {}
    for aroma in dict:
        train = binary_classify_conversion(train_classes, aroma)
        train = train.fillna(0)
        train_features = train_features.fillna(0)

        dev = binary_classify_conversion(dev_classes, aroma)
        dev = dev.fillna(0)
        dev_features = dev_features.fillna(0)


        min_max_scaler = preprocessing.MinMaxScaler()
        scaled = min_max_scaler.fit_transform(train_features)

        # scaled = preprocessing.normalize(train_features, norm='l1')

        model = LogisticRegression(penalty="l1", solver="liblinear")
        # model = KNeighborsClassifier(n_neighbors=5, weights="distance")
        train_features = train_features.rename_axis(None)
        model.fit(scaled, train.values.ravel())
        model_dict[aroma] = model

    return model_dict


def binary_classify_conversion(classes_, in_class):
    classes = classes_.copy()
    for index, row in classes.iterrows():
        # arr = get_array_from_string(row["Aromas"])
        for entry in row["Aromas"]:
            if entry == in_class:
                classes.at[index, "Aromas"] = 1
                break
        if classes.at[index, "Aromas"] != 1:
            classes.at[index, "Aromas"] = 0

    return classes


def ensemble_prediction(model_dict, dev_features, dev_classes):
    column_names = ['real', 'predicted']
    df = pd.DataFrame(columns=column_names)
    predicted = {}

    dev_classes = dev_classes.to_frame()
    for aroma in model_dict:
        dev = binary_classify_conversion(dev_classes, aroma)
        dev = dev.fillna(0)
        dev_features = dev_features.fillna(0)

        min_max_scaler = preprocessing.MinMaxScaler()
        scaled = min_max_scaler.fit_transform(dev_features)
        # scaled = preprocessing.normalize(dev_features, norm='l1')

        predictions = model_dict[aroma].predict(scaled)
        s = pd.Series(predictions)
        s = s.to_frame()

        correct = 0
        size = 0
        for index, row in s.iterrows():
            if s.iloc[index][0] == 1:
                if aroma in predicted:
                    arr = predicted[aroma]
                    arr.append(index)
                    predicted[aroma] = arr
                else:
                    predicted[aroma] = [index]


            size += 1
            # print(f"{s.iloc[index][0]} and {dev.iloc[index][0]}")

            if s.iloc[index][0] == dev.iloc[index][0]:
                correct += 1

        print(f"{aroma} - accuracy: {correct / size}")

    # print(predicted)
    predicted_reverse = {}
    for aroma in predicted:
        for index in predicted[aroma]:
            if index in predicted_reverse:
                arr = predicted_reverse[index]
                arr.append(aroma)
                predicted_reverse[index] = arr
            else:
                predicted_reverse[index] = [aroma]

    # print(predicted_reverse)
    for index in predicted_reverse:
        real = dev_classes.iloc[index][0]
        predicted = predicted_reverse[index]
        df = df.append({'real': real, 'predicted': predicted}, ignore_index=True)

    return df


def output_results(df):
    total = 0
    partial_correct = 0
    total_correct = 0
    none_correct = 0
    num_correct = 0
    num_total = 0
    for index, row in df.iterrows():
        total += 1
        for aroma in row["real"]:
            num_total += 1
            if aroma in row["predicted"]:
                num_correct += 1

        if row["real"] == row["predicted"]:
            total_correct += 1
        else:
            got_one = False
            for aroma in row["real"]:
                if aroma in row["predicted"]:
                    partial_correct += 1
                    got_one = True
                    break
            if not got_one:
                none_correct += 1

    print(
        f"Completely correct: {total_correct}/{total}\nPartial correct: {partial_correct}/{total}\nNone correct: {none_correct}/{total}")
    print(f"Of {num_total} aromas {num_correct} were correct for an accuracy of {num_correct / num_total}")

def baseline_model_accuracy(aroma_dict, dev_classes, dev_features):
    seed(1)
    column_names = ['real', 'predicted']
    df = pd.DataFrame(columns=column_names)

    total_aromas = 0
    for aroma in aroma_dict:
        total_aromas += aroma_dict[aroma]


    for aroma in aroma_dict:
        numba = round(aroma_dict[aroma]/total_aromas, 2)
        numba = numba *100
        print(f"{aroma}: {numba}%")

    total_correct = 0
    total_incorrect = 0
    for index, row in dev_features.iterrows():
        predicted = []
        for aroma in aroma_dict:
            percentage = aroma_dict[aroma] / total_aromas
            if random() >= percentage:
                predicted.append(aroma)

        for entry in predicted:
            if entry in dev_classes.iloc[index]:
                total_correct += 1
            else:
                total_incorrect += 1

        df = df.append({'real': dev_classes.iloc[index], 'predicted': predicted}, ignore_index=True)

    print(f"Total correct predictions: {total_correct}\nTotal incorrect predictions: {total_incorrect}\nAccuracy: {total_correct/(total_correct+total_incorrect)}")

train_data = pd.read_csv("Data/train_data.csv")
dev_data = pd.read_csv("Data/dev_data.csv")

dev_data = replace_aromas(dev_data)
dev_classes = dev_data.iloc[:, -3]
dev_features = dev_data.iloc[:, :-3]

train_data = replace_aromas(train_data)
train_classes = train_data.iloc[:, -3]
train_features = train_data.iloc[:, :-3]

aroma_dict = get_aroma_dict(train_classes)
dict = train_all_vs_one(train_features, train_classes, dev_features, dev_classes)

print(aroma_dict)
baseline_model_accuracy(aroma_dict, dev_classes, dev_features)
df = ensemble_prediction(dict, dev_features, dev_classes)
output_results(df)