#!/bin/bash

#set -e

source config.sh

echo "Granting root user access to X display"
xhost +si:localuser:root

echo "Running Docker image: ${docker_image_id}"
nvidia-docker run -it --rm \
    --name ${docker_image_name} \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --env="DISPLAY" \
    --device=/dev/dri:/dev/dri \
    ${docker_image_id}


echo "Revoking root user access to X display"
xhost -si:localuser:root
