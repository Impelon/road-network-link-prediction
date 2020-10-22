import sys
import os
import time
import json
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import __init__ as util

PLOT_CLASSIFIERS_ONLY = "classifiers_only"
PLOT_ALL = True
PLOT_NONE = False

reverse_scores_low_auc = True
save_reversed_scores_low_auc = False
plot_durations = False
plot_thresholds = True
plot_updated_network = PLOT_CLASSIFIERS_ONLY
plot_title = True
save_plots = False


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

def evaluate(dataset):
    global dataset_folder
    dataset_folder = dataset

    with open(get_data_filepath("training_network"), "r") as file:
        training_network = util.reconstruct_network_from_dgraph_json(json.load(file))
    with open(get_data_filepath("test_network"), "r") as file:
        test_network = util.reconstruct_network_from_dgraph_json(json.load(file))

    # add all json files in dataset-folder as potential similarity-scores
    similarity_scores = [os.path.splitext(path)[0] for path in os.listdir(dataset_folder) if path.lower().endswith(".json")]
    durations = []
    fig = plt.figure()
    for similarity in similarity_scores:
        data = read_similarity_score(similarity)
        if not data:
            continue
        scored_edges, duration = data
        scores, labels = zip(*((s, test_network.has_edge(u, v)) for (u, v, s) in scored_edges))
        try:
            auc = sk.metrics.roc_auc_score(labels, scores)
        except:
            print("Scores or labels could not be interpreted for similarity-score '{}', going to skip.".format(similarity))
            continue
        if reverse_scores_low_auc and auc < 0.5:
            # reverse scores if auc is lower than 0.5 for training data;
            # this means the opposite labels should be assigned for this similarity-score!
            scored_edges = [(edge[0], edge[1], -edge[2]) for edge in scored_edges]
            scores = [s for (u, v, s) in scored_edges]
            print("reversed scores of similarity-score '{}' with AUC {}.".format(similarity, auc))
            auc = sk.metrics.roc_auc_score(labels, scores)
            print("new AUC is {}.".format(auc))
            if save_reversed_scores_low_auc:
                with open(get_data_filepath(similarity), "w") as file:
                    json.dump({"scored_edges": scored_edges, "duration": duration}, file)
        durations.append(duration)
        print("{0} took {1} seconds = {2} minutes = {3} hours.".format(similarity, duration, duration / 60, duration / 3600))
        fpr, tpr, thresholds = sk.metrics.roc_curve(labels, scores)
        optimal_youden_index = np.argmax(tpr - fpr)
        plot = plt.plot(fpr, tpr, label="{0} (AUC: {1:.5})".format(similarity, auc))
        if plot_thresholds:
            plt.plot((fpr[optimal_youden_index], fpr[optimal_youden_index]), (fpr[optimal_youden_index], tpr[optimal_youden_index]), ":",
                     color=plot[0].get_color(), label="{0} opt. threshold: {1:.5}".format(similarity, float(thresholds[optimal_youden_index])))
    plt.plot((0, 1), (0, 1), 'k--', label="'ideal' random (AUC: 0.5)")
    plt.legend(loc='best')
    plt.xlabel('False positive rate (fpr)')
    plt.ylabel('True positive rate (tpr)')
    plt.title('ROC curve')
    plt.show()
    if save_plots:
        fig.savefig("roc_curve.pdf", bbox_inches="tight", pad_inches=0)

    if plot_durations:
        plt.bar(tuple(range(len(durations))), durations)
        plt.xticks(tuple(range(len(durations))), similarity_scores)
        plt.show()

    if plot_updated_network:
        fig, ax = ox.plot_graph(training_network, edge_linewidth=2, show=False, close=False)
        if plot_title:
            fig.suptitle("training network")
        plt.show()
        if save_plots:
            fig.savefig("training_network.pdf", bbox_inches="tight", pad_inches=0)
        fig, ax = ox.plot_graph(test_network, edge_linewidth=2, show=False, close=False)
        if plot_title:
            fig.suptitle("test network")
        plt.show()
        if save_plots:
            fig.savefig("test_network.pdf", bbox_inches="tight", pad_inches=0)
        combined_edges = None
        if plot_updated_network == PLOT_CLASSIFIERS_ONLY:
            similarity_scores = list(filter(is_classifier, similarity_scores))
        for similarity in similarity_scores:
            data = read_similarity_score(similarity)
            if not data:
                continue
            scored_edges, duration = data
            scores, labels = zip(*((s, test_network.has_edge(u, v)) for (u, v, s) in scored_edges))
            if is_classifier(similarity):
                opt_threshold = 0.5
            else:
                try:
                    fpr, tpr, thresholds = sk.metrics.roc_curve(labels, scores)
                except:
                    continue
                optimal_youden_index = np.argmax(tpr - fpr)
                opt_threshold = thresholds[optimal_youden_index]
            updated_network = training_network.copy()
            new_edges = set((edge[0], edge[1]) for edge in scored_edges if edge[2] >= opt_threshold)
            if not combined_edges:
                combined_edges = new_edges
            else:
                combined_edges = combined_edges.intersection(new_edges)
            updated_network.add_edges_from(new_edges)
            #print(sk.metrics.classification_report(labels, [prediction[2] >= opt_threshold for prediction in json_data["predictions"]]))
            edge_colors = ["green" if test_network.has_edge(*edge) else "gray" if training_network.has_edge(*edge)
                           else "red" for edge in updated_network.edges()]
            fig, ax = ox.plot_graph(updated_network, edge_color=edge_colors, edge_linewidth=2, show=False, close=False)
            if plot_title:
                fig.suptitle(similarity)
            plt.show()
            if save_plots:
                fig.savefig(similarity + ".pdf", bbox_inches="tight", pad_inches=0)
        if combined_edges:
            updated_network = training_network.copy()
            updated_network.add_edges_from(combined_edges)
            edge_colors = ["green" if test_network.has_edge(*edge) else "gray" if training_network.has_edge(*edge)
                           else "red" for edge in updated_network.edges()]
            fig, ax = ox.plot_graph(updated_network, edge_color=edge_colors, edge_linewidth=2, show=False, close=False)
            if plot_title:
                fig.suptitle("Intersection of all edges")
            plt.show()
            if save_plots:
                fig.savefig(similarity + ".pdf", bbox_inches="tight", pad_inches=0)

if __name__ == "__main__":
    if "-h" in sys.argv or "--help" in sys.argv:
        print("A script that evaluates and plots the results of an experiment.")
        print()
        print("python " + sys.argv[0] + " [options] dataset-folder")
        print("Options:")
        print(" -h          Show help.")
        sys.exit(0)
    arguments = tuple(filter(lambda x: not x.startswith("-"), sys.argv[1:]))
    if len(arguments) < 1:
        print("You need to specify a path to a dataset-folder.")
        sys.exit(0)
    evaluate(arguments[0])
