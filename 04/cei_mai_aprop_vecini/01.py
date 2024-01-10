from sklearn import datasets
import math
import operator


def load_dataset():
    iris = datasets.load_iris()
    training_set = iris.data
    test_set = iris.target
    return training_set, test_set


training_set, test_set = load_dataset()
print(f'Train: {repr(len(training_set))}')
print(f'Test: {repr(len(test_set))}')


def euclidian_distance(instance1, instance2, length):
    distance = 0
    for x in range(3):
        distance += pow(instance1[x] - float(instance2[x]), 2)

    return math.sqrt(distance)


def get_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance)
    for x in range(len(training_set)):
        dist = euclidian_distance(test_instance[x], training_set[x-1], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))

    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])

    return neighbors


def get_response(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1

        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def get_accuracy(test_set1, predictions):
    correct = 0
    for x in range(len(test_set1)):
        if test_set1[x] == predictions[x]:
            correct += 1

    return correct / float(len(test_set1)) * 100




def main():
    training_set, test_set = load_dataset()
    print(f'Train: {repr(len(training_set))}')
    print(f'Test: {repr(len(test_set))}')

    predictions = []
    k = 3
    for x in range(len(test_set)):
        neighbors = get_neighbors(training_set, training_set, 3)
        result = get_response(neighbors)
        predictions.append(result)
        print(f'Predicted {repr(result)}, actual = {test_set[x]}')
    accuracy = get_accuracy(test_set, predictions)
    print(f"Accuracy: {repr(accuracy)}%")


if __name__ == '__main__':
    main()
