#!/usr/bin/env bash

# This script provides some functionality for controlling a
# dgraph-standalone docker-image.
# This image needs to be installed first:
# 1. docker pull dgraph/dgraph:v20.03.0
# 2. mkdir -p ~/dgraph
# 3. docker run -it -p 5080:5080 -p 6080:6080 -p 8080:8080 -p 9080:9080 -p 8000:8000 -v ~/dgraph:/dgraph --name dgraph dgraph/dgraph:v20.03.0 dgraph zero

DGRAPH_CONTAINER_NAME="dgraph"

is_running() {
  return $([ "$(sudo docker container inspect -f '{{.State.Status}}' $DGRAPH_CONTAINER_NAME)" == "running" ])
}

toggle() {
  if is_running; then
    stop
  else
    start
  fi
}

stop() {
  echo "Stopping dgraph standalone docker-image..."
  sudo docker stop dgraph
  echo "Done."
}

start() {
  echo "Starting dgraph standalone docker-image..."
  sudo docker start dgraph
  echo "Starting dgraph alpha..."
  sudo docker exec -it -d dgraph dgraph alpha --lru_mb 2048 --zero localhost:5080
  echo "Starting dgraph-ratel..."
  sudo docker exec -it -d dgraph dgraph-ratel
  echo "Done."
}

load_dataset() {
  if [[ $# < 1 ]]; then
    echo "You need to specify the path for the dataset"
    exit 1
  fi
  format=rdf
  if [[ $# > 1 ]]; then
    format=$2
  fi
  echo "Starting loading dataset from path '$1'..."
  sudo docker exec -it dgraph dgraph live -f "$1" --format=$format --alpha localhost:9080 --zero localhost:5080 -c 1
  echo "Done."
}

if [[ $# < 1 ]]; then
  toggle
  exit 0
fi

case $1 in
  status) is_running;;
  start) start;;
  stop) stop;;
  toggle) toggle;;
  load) load_dataset $2 $3;;
  *) echo "Unknown action, try one of: start, stop, toggle, load";;
esac
