#!/bin/bash

docker run --rm -it -d --shm-size=64gb --gpus all \
    -v /your_path_to_repo/RCTrans:/home/docker_rctrans/RCTrans \
    -v /your_path_to_data:/home/docker_rctrans/HPR3 \
    --name rctrans  rctrans:latest "/bin/bash"
