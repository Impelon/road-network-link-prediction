# About

In this proof-of-concept project ideas of link prediction were applied to try to predict which points in a real-world road network should be connected.
Link prediction is a discipline originating from social networks, and not a lot of previous work experimented with applying its methods to road networks.
Missing roads would be predicted using different features and a random forest classifier trained for a given network.
Check out the [seminar paper](writing/paper/paper.pdf) for more details on the approach and results.

## Installation

1.  clone the repository via
    ```
    git clone https://github.com/Impelon/road-network-link-prediction.git
    ```
2.  optionally use `virtualenv` to make a virtual environment for python-packages
3.  open `code`-folder in terminal
4.  install dependencies via
    ```
    pip3 install -r pip-requirements.txt
    ```
5.  The project uses Dgraph to demonstrate compatability with industry-grade graph databases.
    This also allows storing multiple road networks in the same database. For some simpler functionalities Dgraph is not required, but the link prediction assumes a draph instance is running.
    You can install it as a docker container:
    1. `docker pull dgraph/dgraph:v20.03.0`
    2. `mkdir -p ~/dgraph`
    3. `docker run -it -p 5080:5080 -p 6080:6080 -p 8080:8080 -p 9080:9080 -p 8000:8000 -v ~/dgraph:/dgraph --name dgraph dgraph/dgraph:v20.03.0 dgraph zero`

## Usage

The python scripts in the [code](code) folder provide a small usage header, please consult them for further details by supplying a `--help` flag.

1.  In general the workflow starts by staring the Dgraph database, here via the docker image:
    ```
    ./control_dgraph_docker.sh start
    ```
2.  Import a simplified road network into the database:
    ```
    python3 network_to_dgraph_json.py -a -d "Technische Universität Berlin, Berlin"
    ```
3.  Optionally render the imported graph:
    ```
    python3 render_graph.py "Technische Universität Berlin, Berlin"
    ```
4.  Collect features from the graph and train some classifiers, saving the trained classifier in the given location:
    ```
    python3 predict_links.py data-and-model-folder "Technische Universität Berlin, Berlin"
    ```
5.  Evaluate link prediction similarity scores, including the trained classifiers:
    ```
    python3 evaluate_link_prediction.py data-and-model-folder
    ```

A trained classifer can also be used to predict roads on an entirely new graph using `classifier_predict_links.py`.
