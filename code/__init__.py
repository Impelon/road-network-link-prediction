import json
import random
import osmnx as ox
import networkx as nx

__all__ = ["network_to_dgraph_json", "reconstruct_network_from_dgraph_json", "load_nodes_from_dgraph"]


def network_to_dgraph_json(network):
    """
    Extract important data from the given road-network and produce
    JSON(-like objects) that can be used directly in dgraph-mutations.

    Keep in mind that nodes and edges can have multiple (OSM-)IDs if they are created by
    merging multiple nodes/edges;
    therefore in the JSON the osmids-attribute encodes the IDs (or ID) as a string:
    osm-id = 1 -> osmids-attribute = "1"
    osm-id = [1, 2] -> osmids-attribute = "[1, 2]"

    Also for the same reason two nodes can be connected by multiple edges;
    therefore in the JSON the osmids-attribute for the "connects_to"-relation encodes
    these multiple edges (or their IDs) as as a string:
    (edge: 0, osm-id: 1) -> osmids-attribute = "1"
    (edge: 0, osm-id: [1, 2]) -> osmids-attribute = "[1, 2]"
    (edge: 0, osm-id: 1), (edge: 1, osm-id: 3) -> osmids-attribute = "1:3"
    (edge: 0, osm-id: [1, 2]), (edge: 1, osm-id: 3) -> osmids-attribute = "[1, 2]:3"
    (edge: 0, osm-id: [1, 2]), (edge: 1, osm-id: 3), (edge: 2, osm-id: 5) -> osmids-attribute = "[1, 2]:3:5"
    """

    uid_prefix = "_:"
    node_uid_prefix = uid_prefix + "node"

    json_mutations = []
    for node, attributes in network.nodes.items():
        json_mutations.append({
            "uid": node_uid_prefix + str(attributes.get("osmid", attributes.get("osmids"))),
            "osmids": str(attributes.get("osmid", attributes.get("osmids"))),
            "location": {
                "type": "Point",
                "coordinates": [
                    attributes["x"],
                    attributes["y"]
                ]
            }
        })
    for node, neighbors in network.adj.items():
        for neighbor, edges in neighbors.items():
            json_mutations.append({
                "uid": node_uid_prefix + str(network.nodes[node].get("osmid", network.nodes[node].get("osmids"))),
                "connects_to": {
                    "uid": node_uid_prefix + str(network.nodes[neighbor].get("osmid", network.nodes[neighbor].get("osmids"))),
                    "connects_to|osmids": ":".join(str(edge_attributes.get("osmid", edge_attributes.get("osmids"))) for edge, edge_attributes in edges.items() if "osmid" in edge_attributes or "osmids" in edge_attributes)
                }
            })
    return json_mutations


def reconstruct_network_from_dgraph_json(data):
    """
    Constructs a graph containing the given road-network-data that can be rendered with OSMnx.
    """
    G = nx.MultiDiGraph(crs=ox.settings.default_crs)
    for node in data:
        if "location" in node:
            attributes = node.copy()
            attributes["x"] = attributes["location"]["coordinates"][0]
            attributes["y"] = attributes["location"]["coordinates"][1]
            attributes.pop("location", 0)
            attributes.pop("connects_to", 0)
            G.add_node(node["uid"], **attributes)
    for node in data:
        if "connects_to" in node:
            node_uid = node["uid"]
            if isinstance(node["connects_to"], list):
                for neighbor in node["connects_to"]:
                    neighbor_uid = neighbor["uid"]
                    if neighbor_uid in G.nodes:
                        G.add_edge(node_uid, neighbor_uid)
            else:
                neighbor_uid = node["connects_to"]["uid"]
                if neighbor_uid in G.nodes:
                    G.add_edge(node_uid, neighbor_uid)
    return G


def load_json_from_dgraph(latitude, longitude, radius=None):
    """
    Loads road-network-data for the given coordinates from the local dgraph-database
    which can be used to reconstruct the network as a graph.
    """
    import pydgraph
    if not radius:
        radius = 3000

    client_stub = pydgraph.DgraphClientStub('localhost:9080')
    client = pydgraph.DgraphClient(client_stub)

    query = """query nodes($coordinates: string, $radius: float) {
      nodes(func: near(location, $coordinates, $radius))
      {
        uid,
        osmids,
        location,
        connects_to {
            uid
        }
      }
    }"""
    variables = {
        "$coordinates": "[{},{}]".format(longitude, latitude),
        "$radius": str(radius)
    }

    txn = client.txn(read_only=True)
    try:
        result = txn.query(query, variables=variables)
    finally:
        txn.discard()
    client_stub.close()
    return json.loads(result.json)["nodes"]

def to_simple_edge_network(network):
    """
    Converts a MultiDiGraph to a DiGraph. This is not lossless.
    """
    simple_network = nx.DiGraph(crs=network.graph["crs"])
    for node in network.nodes:
        simple_network.add_node(node, **network.nodes[node])
    for node, neighbors in network.adjacency():
        for neighbor in neighbors:
            simple_network.add_edge(node, neighbor)
    return simple_network


def simplify_network(network):
    """
    Simplifies a given road-network using OSMnx.
    """
    network_proj = ox.project_graph(network)
    simplified_proj = ox.consolidate_intersections(network_proj, rebuild_graph=True, tolerance=50, dead_ends=False)
    simplified = ox.project_graph(simplified_proj, network.graph["crs"])
    return simplified


def split_test_from_training_data(network, ratio=0.1):
    """
    Splits a given network into two networks with disjoint edges randomly.
    """
    test_edges = random.sample(list(network.edges()), int(len(network.edges()) * ratio))
    training_network = network.copy()
    training_network.remove_edges_from(test_edges)
    test_network = network.copy()
    test_network.remove_edges_from(training_network.edges())
    return training_network, test_network
