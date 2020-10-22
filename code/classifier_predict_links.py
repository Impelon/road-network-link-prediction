import sys
import os
import json
import joblib
import time
import networkx as nx
import sklearn as sk
import numpy as np
import __init__ as util
from sklearn.ensemble import RandomForestClassifier

def get_data_filepath(name):
    return os.path.join(dataset_folder, name + ".json")

def read_similarity_score(name):
    filepath = get_data_filepath(name)
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r") as file:
            json_data = json.load(file)
        return json_data["scored_edges"], json_data["duration"]
    except Exception as ex:
        print("Exception occurred while accessing '{}' as similarity_score: {}".format(name , ex))
        return None

def predict_links_with_classifier(classifier, dataset, output_name):
    global dataset_folder
    dataset_folder = dataset

    similarity_scores = ["jaccard_coefficient", "adamic_adar_index", "common_neighbors",
                         "k_shortest_paths_non_aggregated_0", "k_shortest_paths_non_aggregated_1", "k_shortest_paths_non_aggregated_2",
                         "k_shortest_paths", "min_and_max_neighbors_0", "min_and_max_neighbors_1",
                         "geospacial_distances_0", "geospacial_distances_1", "geospacial_distances_2"]
    features = []
    for similarity in similarity_scores:
        data = read_similarity_score(similarity)
        if not data:
            continue
        scored_edges, _ = data
        scores = [s for (u, v, s) in scored_edges]
        features.append(scores)
    features = np.array(features)
    features = features.T

    with open(get_data_filepath("training_network"), "r") as file:
        training_network = util.reconstruct_network_from_dgraph_json(json.load(file))
    training_network = util.to_simple_edge_network(training_network).to_undirected(reciprocal=False)
    complement_edges = list(nx.complement(training_network).edges())

    classifier = joblib.load(classifier)

    print("Starting with {}.".format(output_name))
    start_time = time.time()
    scores = classifier.predict_proba(features)
    scores = list(map(lambda x: float(x[1]), scores))
    scored_edges = [(*edge, score) for edge, score in zip(complement_edges, scores)]
    end_time = time.time()
    duration = end_time - start_time
    print("Done with {}. Took {} in total.".format(output_name, time.strftime("%X", time.gmtime(duration))))
    with open(os.path.join(dataset_folder, output_name + ".json"), "w") as file:
        json.dump({"scored_edges": scored_edges, "duration": duration}, file)

if __name__ == "__main__":
    if "-h" in sys.argv or "--help" in sys.argv:
        print("A script that loads a trained classifier and features from a dataset.",
              "The script will use the classifier to predict the classes for the given features",
              "and save the results.")
        print()
        print("python " + sys.argv[0] + " [options] classifier-file dataset-folder result-name")
        print("Options:")
        print(" -h          Show help.")
        sys.exit(0)
    arguments = tuple(filter(lambda x: not x.startswith("-"), sys.argv[1:]))
    if len(arguments) < 3:
        print("You need to specify a path to a classifier saved with joblib,",
              "a path to a dataset-folder and a name for the results to be saved as.")
        sys.exit(0)
    predict_links_with_classifier(arguments[0], arguments[1], arguments[2])
