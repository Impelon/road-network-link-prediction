import sys
import json
import osmnx as ox
import __init__ as util


def network_from_location(location=None, is_place=False, **kwargs):
    """
    Load road-network-data from OpenStreetMap (OSM) via OSMnx.

    The network structure will be simplified by default, see:
    https://github.com/gboeing/osmnx-examples/blob/master/notebooks/04-simplify-graph-consolidate-nodes.ipynb
    """
    if not location:
        location = "TU Berlin, Berlin, Deutschland"
    if is_place:
        return ox.graph_from_place(location, **kwargs)
    return ox.graph_from_address(location, **kwargs)


if __name__ == "__main__":
    access_dgraph = False
    print_json = True
    draw_before = False
    is_place = False

    if "-h" in sys.argv or "--help" in sys.argv:
        print("A script that loads data from OpenStreetMap and extracts relevant",
              "information as JSON-data that can be used directly in dgraph-mutations",
              "and prints it out.")
        print()
        print("python " + sys.argv[0] + " [options] address [distance]")
        print("or")
        print("python " + sys.argv[0] + " -x [options] place")
        print("Options:")
        print(" -h          Show help.")
        print(" -x          Interpret the given location as a place instead of an address, therefore fetching only data relevant to that area.")
        print(" -a          Access the local dgraph-database and upload the network; disables printing of the data.")
        print(" -p          Enables printing the data to the standard output.")
        print(" -d          Draw the graph before doing the conversion. Ask whether to proceed with the conversion.")
        sys.exit(0)
    if "-x" in sys.argv or "--place" in sys.argv:
        is_place = True
    if "-a" in sys.argv or "--access" in sys.argv:
        access_dgraph = True
        print_json = False
    if "-p" in sys.argv or "--print" in sys.argv:
        print_json = True
    if "-d" in sys.argv or "--draw" in sys.argv:
        draw_before = True

    arguments = tuple(filter(lambda x: not x.startswith("-"), sys.argv[1:]))
    location = None
    network_arguments = {"network_type": "drive"}
    if len(arguments) > 0:
        location = arguments[0]
    if len(arguments) > 1 and not is_place:
        network_arguments["dist"] = int(arguments[1])

    network = network_from_location(location, is_place, **network_arguments)
    if draw_before:
        fig, ax = ox.plot_graph(network, edge_linewidth=2)
        print("Type 0 to abort or 1 to proceed.", file=sys.stderr)
        while True:
            answer = str(input(""))
            if answer == "0":
                print("Aborted.", file=sys.stderr)
                sys.exit(1)
            elif answer == "1":
                break
    json_mutations = util.network_to_dgraph_json(network)

    if print_json:
        print(json.dumps(json_mutations, indent=2))
    if access_dgraph:
        import pydgraph
        # open connection to local dgraph-database
        client_stub = pydgraph.DgraphClientStub('localhost:9080')
        client = pydgraph.DgraphClient(client_stub)

        # set schema for the data about to be loaded
        schema = """<connects_to>: [uid] .
        <location>: geo @index(geo) .
        <osmids>: string .
        """
        op = pydgraph.Operation(schema=schema)
        client.alter(op)

        # transfer data in transaction
        txn = client.txn()
        try:
            txn.mutate(set_obj=json_mutations)
            txn.commit()
        finally:
            txn.discard()
        client_stub.close()
