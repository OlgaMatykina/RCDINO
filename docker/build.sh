#!/bin/bash

echo "Building container"
docker build . \
    -f Dockerfile \
    -t rctrans:latest \
    --progress plain 

