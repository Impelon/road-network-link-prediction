import sys
import os
import time
import itertools
import json
import joblib
import osmnx as ox
import networkx as nx
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import __init__ as util
from sklearn.ensemble import RandomForestClassifier


# feature-function / similarity-scores

def common_neighbors(G, ebunch=None):
    if not ebunch:
        ebunch = nx.complement(G).edges()
    return ((u, v, len(tuple(nx.common_neighbors(G, u, v)))) for u, v in ebunch)


def k_shortest_paths_for_edge_non_aggregated(G, u, v, k=3):
    try:
        paths = nx.shortest_simple_paths(G, u, v)
        shortest_paths = tuple(map(len, itertools.islice(paths, k)))
    except:
        shortest_paths = ()
    return shortest_paths + tuple(itertools.repeat(10000000, k - len(shortest_paths)))

def k_shortest_paths_non_aggregated(G, ebunch=None, k=3):
    if not ebunch:
        ebunch = nx.complement(G).edges()
    return ((u, v, k_shortest_paths_for_edge_non_aggregated(G, u, v, k=k)) for u, v in ebunch)


def k_shortest_paths_for_edge(G, u, v, k=3):
    return sum(k_shortest_paths_for_edge_non_aggregated(G, u, v, k=k))


def k_shortest_paths(G, ebunch=None, k=3):
    if not ebunch:
        ebunch = nx.complement(G).edges()
    return ((u, v, k_shortest_paths_for_edge(G, u, v, k=k)) for u, v in ebunch)


def min_and_max_neighbors(G, ebunch=None):
    if not ebunch:
        ebunch = nx.complement(G).edges()
    return ((u, v, sorted((len(tuple(G.neighbors(u))), len(tuple(G.neighbors(v)))))) for u, v in ebunch)


def geospacial_distances_for_edge(G, u, v):
    x = abs(G.nodes[u]["x"] - G.nodes[v]["x"])
    y = abs(G.nodes[u]["y"] - G.nodes[v]["y"])
    return x, y, np.math.hypot(x, y)


def geospacial_distances(G, ebunch=None):
    if not ebunch:
        ebunch = nx.complement(G).edges()
    return ((u, v, geospacial_distances_for_edge(G, u, v)) for u, v in ebunch)


# helper-functions

def gather_feature(feature_function, network, edges):
    print("Starting with feature '{}'.".format(feature_function.__name__))
    start_time = time.time()
    scored_edges = list(feature_function(network, ebunch=edges))
    scores = [s for (u, v, s) in scored_edges]
    end_time = time.time()
    duration = end_time - start_time
    print("Done with feature '{}'. Took {}.".format(feature_function.__name__, time.strftime("%X", time.gmtime(duration))))
    return scored_edges, scores, duration


def gather_features_and_labels(network, edges, feature_functions, save_features=False):
    features = []
    for feature_function in feature_functions:
        scored_edges, scores, duration = gather_feature(feature_function, network, edges)
        if not scores:
            continue
        try:
            for i in range(len(scores[0])):
                sliced_scored_edges = list(map(lambda x: (x[0], x[1], x[2][i]), scored_edges))
                sliced_scores = [s for (u, v, s) in sliced_scored_edges]
                features.append(sliced_scores)
                if save_features:
                    save_feature("{}_{}".format(feature_function.__name__, i), sliced_scored_edges, duration)
        except:
            features.append(scores)
            if save_features:
                save_feature(feature_function.__name__, scored_edges, duration)

        if feature_function == k_shortest_paths_non_aggregated:
            start_time = time.time()
            scored_edges = list(map(lambda x: (x[0], x[1], sum(x[2])), scored_edges))
            scores = [s for (u, v, s) in scored_edges]
            end_time = time.time()
            duration += end_time - start_time
            features.append(scores)
            if save_features:
                save_feature("k_shortest_paths", scored_edges, duration)
    features = np.array(features)
    features = features.T
    labels = [network.has_edge(*edge) for edge in edges]
    return features, labels


def save_feature(name, scored_edges, duration):
    with open(os.path.join(dataset_folder, name + ".json"), "w") as file:
        json.dump({"scored_edges": scored_edges, "duration": duration}, file)


# main script

def predict_links(dataset, latitude, longitude, radius=None):
    global dataset_folder
    dataset_folder = dataset
    if radius == None:
        radius = 5000
    data = util.load_json_from_dgraph(latitude, longitude, radius)
    original_network = util.reconstruct_network_from_dgraph_json(data)
    # Scores only work on undirected, non-multi-edge graphs!
    # The dgraph-representation already compresses multiple edges into a single edge,
    # so this representation does not loose much information.
    original_network = util.simplify_network(original_network)
    original_network = util.to_simple_edge_network(original_network).to_undirected(reciprocal=False)
    training_network, test_network = util.split_test_from_training_data(original_network, ratio=0.3)

    os.makedirs(dataset_folder, exist_ok=True)
    with open(os.path.join(dataset_folder, "training_network.json"), "w") as file:
        dgraph_representation = util.network_to_dgraph_json(training_network)
        json.dump(dgraph_representation, file)
        # simplification replaces original node-indices
        # for consistent indexing, the networks are restored from their dgraph-representation
        training_network = util.reconstruct_network_from_dgraph_json(dgraph_representation)
    with open(os.path.join(dataset_folder, "test_network.json"), "w") as file:
        dgraph_representation = util.network_to_dgraph_json(test_network)
        json.dump(dgraph_representation, file)
        # simplification replaces original node-indices
        # for consistent indexing, the networks are restored from their dgraph-representation
        test_network = util.reconstruct_network_from_dgraph_json(dgraph_representation)

    # ox.plot_graph(training_network, edge_linewidth=2)
    # ox.plot_graph(test_network, edge_linewidth=2)

    training_network = util.to_simple_edge_network(training_network).to_undirected(reciprocal=False)
    complement_edges = list(nx.complement(training_network).edges())
    similarity_scores = [nx.jaccard_coefficient, nx.adamic_adar_index, common_neighbors,
                         k_shortest_paths_non_aggregated, min_and_max_neighbors, geospacial_distances]

    features, labels = gather_features_and_labels(training_network, complement_edges, similarity_scores, save_features=True)

    classifier_small = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0)
    classifier_large = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=0)

    training_edges = list(training_network.edges()) + complement_edges
    print("Starting with random-forest-classifiers.")
    start_time = time.time()
    training_features, _ = gather_features_and_labels(training_network, training_edges, similarity_scores)
    training_labels = [training_network.has_edge(*edge) or test_network.has_edge(*edge) for edge in training_edges]
    end_time = time.time()
    duration = end_time - start_time
    print("Done with gathering additional features. Took {}.".format(time.strftime("%X", time.gmtime(duration))))

    print("Starting with training small random-forest-classifier.")
    start_time = time.time()
    classifier_small.fit(training_features, training_labels)
    end_time = time.time()
    duration_training_small = end_time - start_time
    print("Done with training small random-forest-classifier. Took {}.".format(time.strftime("%X", time.gmtime(duration_training_small))))
    print("feature importances:", classifier_small.feature_importances_)
    scores = classifier_small.predict_proba(features)
    scores = list(map(lambda x: float(x[1]), scores))
    scored_edges = [(*edge, score) for edge, score in zip(complement_edges, scores)]
    end_time = time.time()
    duration += end_time - start_time + duration_training_small
    print("Done with small random-forest-classifier. Took {} in total.".format(time.strftime("%X", time.gmtime(duration))))
    save_feature("random_forest_classifier_small", scored_edges, duration)
    # save classifier model
    joblib.dump(classifier_small, os.path.join(dataset_folder, "random_forest_classifier_small.joblib"))

    print("Starting with training large random-forest-classifier.")
    start_time = time.time()
    classifier_large.fit(training_features, training_labels)
    end_time = time.time()
    duration_training_large = end_time - start_time
    print("Done with training large random-forest-classifier. Took {}.".format(time.strftime("%X", time.gmtime(duration_training_large))))
    print("feature importances:", classifier_large.feature_importances_)
    scores = classifier_large.predict_proba(features)
    scores = list(map(lambda x: float(x[1]), scores))
    scored_edges = [(*edge, score) for edge, score in zip(complement_edges, scores)]
    end_time = time.time()
    duration += end_time - start_time + duration_training_large
    print("Done with large random-forest-classifier. Took {} in total.".format(time.strftime("%X", time.gmtime(duration))))
    save_feature("random_forest_classifier_large", scored_edges, duration)
    # save classifier model
    joblib.dump(classifier_large, os.path.join(dataset_folder, "random_forest_classifier_large.joblib"))


if __name__ == "__main__":
    use_coordinates = False
    if "-h" in sys.argv or "--help" in sys.argv:
        print("A script that computes different similarity scores and classifiers",
              "for link prediction on the given dataset accessed from dgraph.")
        print()
        print("python " + sys.argv[0] + " [options] target-folder address/place [radius]")
        print("or")
        print("python " + sys.argv[0] + " -c [options] target-folder latitude longitude [radius]")
        print("Options:")
        print(" -h          Show help.")
        print(" -c          Use specified coordinates in stead of an address/place.")
        sys.exit(0)
    if "-c" in sys.argv or "--coordinates" in sys.argv:
        use_coordinates = True
    arguments = tuple(filter(lambda x: not x.startswith("-"), sys.argv[1:]))
    latitude = None
    longitude = None
    radius = None
    if len(arguments) < 1:
        print("You need to specify a folder where to store the results for this dataset.")
        sys.exit(1)
    if use_coordinates:
        if len(arguments) < 3:
            print("You need to specify a longitude and latitude.")
            sys.exit(1)
        if len(arguments) > 3:
            radius = arguments[3]
        latitude = arguments[1]
        longitude = arguments[2]
    else:
        if len(arguments) < 2:
            print("You need to specify an address or place.")
            sys.exit(1)
        if len(arguments) > 2:
            radius = arguments[2]
        latitude, longitude = ox.geocode(arguments[1])
    predict_links(arguments[0], latitude, longitude, radius=radius)
