import sys
import os
import json
import itertools
import joblib
import sklearn as sk
import numpy as np
import __init__ as util
from sklearn.metrics import accuracy_score, recall_score, precision_score


def get_group_key(entry):
    filepath = entry[0]
    head, tail = os.path.split(filepath)
    return os.path.split(head)[0] + tail


def is_classifier(name):
    return "classifier" in name.lower()


def get_data_filepath(name):
    if not (name.startswith(".") or os.path.splitext(name)[1]):
        name += ".json"
    return os.path.join(dataset_folder, name)


def read_similarity_score(name):
    filepath = get_data_filepath(name)
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r") as file:
            json_data = json.load(file)
        return json_data["scored_edges"], json_data["duration"]
    except Exception as ex:
        print("Exception occurred while accessing '{}' as similarity_score: {}".format(name, ex))
        return None


def collect(experiments_path):
    global dataset_folder

    auc = {}
    duration = {}
    accuracy = {}
    recall = {}
    precision = {}
    feature_importance = {}

    for root, dirs, files in os.walk(experiments_path):
        dataset_folder = root
        for path in files:
            if path.lower().endswith(".joblib"):
                classifier = joblib.load(get_data_filepath(path))
                feature_importance[get_data_filepath(path)] = classifier.feature_importances_

        similarity_scores = [os.path.splitext(path)[0] for path in files if path.lower().endswith(".json")]
        if not (similarity_scores and os.path.isfile(get_data_filepath("test_network"))):
            continue

        with open(get_data_filepath("test_network"), "r") as file:
            test_network = util.reconstruct_network_from_dgraph_json(json.load(file))

        for similarity in similarity_scores:
            data = read_similarity_score(similarity)
            if not data:
                continue
            scored_edges, similarity_duration = data
            duration[get_data_filepath(similarity)] = similarity_duration
            scores, labels = zip(*((s, test_network.has_edge(u, v)) for (u, v, s) in scored_edges))
            similarity_auc = sk.metrics.roc_auc_score(labels, scores)
            if similarity_auc < 0.5:
                similarity_auc = -similarity_auc + 1
            auc[get_data_filepath(similarity)] = similarity_auc

            if is_classifier(similarity):
                accuracy[get_data_filepath(similarity)] = accuracy_score(labels, [score > 0.5 for score in scores])
                recall[get_data_filepath(similarity)] = recall_score(labels, [score > 0.5 for score in scores])
                precision[get_data_filepath(similarity)] = precision_score(labels, [score > 0.5 for score in scores])

    print("AUC:")
    for key, group in itertools.groupby(sorted(auc.items(), key=get_group_key), get_group_key):
        values = [value for key, value in group]
        print("{}\t\tmean: {:.6f}\t\tstd: {:.6f}\t\tvalues: {}".format(key, np.mean(values), np.std(values), values))
    print("Accuracy:")
    for key, group in itertools.groupby(sorted(accuracy.items(), key=get_group_key), get_group_key):
        values = [value for key, value in group]
        print("{}\t\tmean: {:.6f}\t\tstd: {:.6f}\t\tvalues: {}".format(key, np.mean(values), np.std(values), values))
    print("Recall:")
    for key, group in itertools.groupby(sorted(recall.items(), key=get_group_key), get_group_key):
        values = [value for key, value in group]
        print("{}\t\tmean: {:.6f}\t\tstd: {:.6f}\t\tvalues: {}".format(key, np.mean(values), np.std(values), values))
    print("Precision:")
    for key, group in itertools.groupby(sorted(precision.items(), key=get_group_key), get_group_key):
        values = [value for key, value in group]
        print("{}\t\tmean: {:.6f}\t\tstd: {:.6f}\t\tvalues: {}".format(key, np.mean(values), np.std(values), values))
    print("Feature-importance:")
    for key, group in itertools.groupby(sorted(feature_importance.items(), key=get_group_key), get_group_key):
        values = [value for key, value in group]
        print("{}\t\tmean: {}\t\tstd: {}".format(key, np.mean(values, axis=0), np.std(values, axis=0)))


if __name__ == "__main__":
    experiments_path = "experiments"
    collect(experiments_path)
