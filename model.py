# K-Nearest Neighbors (KNN) Classifier from Scratch
# Iris Flower Classification

import math
import random
import csv

# -----------------------------
# Load Dataset
# -----------------------------
def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append([float(x) for x in row[:-1]] + [row[-1]])
    return dataset

# -----------------------------
# Split dataset
# -----------------------------
def train_test_split(dataset, split_ratio=0.7):
    train_size = int(len(dataset) * split_ratio)
    train_set = random.sample(dataset, train_size)
    test_set = [row for row in dataset if row not in train_set]
    return train_set, test_set

# -----------------------------
# Euclidean Distance
# -----------------------------
def euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)

# -----------------------------
# Neighbors
# -----------------------------
def get_neighbors(train, test_row, k):
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

# -----------------------------
# Prediction
# -----------------------------
def predict_classification(train, test_row, k):
    neighbors = get_neighbors(train, test_row, k)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# -----------------------------
# Accuracy
# -----------------------------
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return (correct / float(len(actual))) * 100.0

# -----------------------------
# Main
# -----------------------------
def main():
    filename = "iris.csv"
    dataset = load_dataset(filename)

    train, test = train_test_split(dataset, 0.7)

    predictions = []
    actual = []

    k = 3

    for row in test:
        output = predict_classification(train, row, k)
        predictions.append(output)
        actual.append(row[-1])
        print(f"Predicted: {output}, Actual: {row[-1]}")

    acc = accuracy_metric(actual, predictions)
    print("\nAccuracy: {:.2f}%".format(acc))
if __name__ == "__main__":
    main()

