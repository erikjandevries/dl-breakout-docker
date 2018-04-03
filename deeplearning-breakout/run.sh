#!/bin/bash

#set -e

command=$1

source config.sh

echo "Granting root user access to X display"
xhost +si:localuser:root

echo "Running Docker image: ${docker_image_id}"
nvidia-docker run -it --rm \
    --name ${docker_image_name} \
    -p 6006:6006 \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --env "DISPLAY" \
    --device /dev/dri:/dev/dri \
    --volume /home/erik/Repositories/GitHub/dl-breakout-docker/deeplearning-breakout/data:/repositories/breakout/data \
    ${docker_image_id} ${command}


echo "Revoking root user access to X display"
xhost -si:localuser:root
