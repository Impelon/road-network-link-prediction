import sys
import json
import osmnx as ox
import networkx as nx
import __init__ as util


if __name__ == "__main__":
    use_coordinates = False
    read_stdin = False
    do_simplification = False
    save_render = False
    if "-h" in sys.argv or "--help" in sys.argv:
        print("A script that renders the reconstructed network from JSON-data. ",
              "The road-network-data can be loaded from the local dgraph-database.")
        print()
        print("python " + sys.argv[0] + " [options] address/place [radius]")
        print("or")
        print("python " + sys.argv[0] + " -c [options] latitude longitude [radius]")
        print("or")
        print("python " + sys.argv[0] + " -r [options]")
        print("Options:")
        print(" -h          Show help.")
        print(" -c          Use specified coordinates in stead of an address/place.")
        print(" -r          Read JSON-data from the standard input instead of reading from the dgraph-database.")
        print(" -y          Simplify graph before rendering.")
        print(" -s          Save rendered result.")
        sys.exit(0)
    if "-c" in sys.argv or "--coordinates" in sys.argv:
        use_coordinates = True
    if "-r" in sys.argv or "--read-stdin" in sys.argv:
        read_stdin = True
    if "-y" in sys.argv or "--simplify" in sys.argv:
        do_simplification = True
    if "-s" in sys.argv or "--save" in sys.argv:
        save_render = True

    if not read_stdin:
        arguments = tuple(filter(lambda x: not x.startswith("-"), sys.argv[1:]))
        latitude = None
        longitude = None
        radius = None
        if use_coordinates:
            if len(arguments) < 2:
                print("You need to specify a longitude and latitude.")
                sys.exit(1)
            if len(arguments) > 2:
                radius = arguments[2]
            latitude = arguments[0]
            longitude = arguments[1]
        else:
            if len(arguments) < 1:
                print("You need to specify an address or place.")
                sys.exit(1)
            if len(arguments) > 1:
                radius = arguments[1]
            latitude, longitude = ox.geocode(arguments[0])
        data = util.load_json_from_dgraph(latitude, longitude, radius)
    else:
        data = json.load(sys.stdin)

    G = util.reconstruct_network_from_dgraph_json(data)
    if do_simplification:
        G = util.simplify_network(G)
    print("Network has {} nodes and {} edges. Clustering coefficient is {}.".format(sum(1 for _ in G.nodes()), sum(1 for _ in G.edges()), nx.average_clustering(util.to_simple_edge_network(G).to_undirected(reciprocal=False))))
    fig, ax = ox.plot_graph(G, edge_linewidth=2)
    if save_render:
        fig.savefig("rendered_{}_{}_{}.pdf".format(latitude, longitude, radius), bbox_inches="tight", pad_inches=0)
