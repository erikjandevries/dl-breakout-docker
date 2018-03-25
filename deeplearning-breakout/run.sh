#!/bin/bash

#set -e

source config.sh

nvidia-docker run -it --rm \
    --name ${docker_image_name} \
    ${docker_image}

#    -v /tmp/.X11-unix:/tmp/.X11-unix \  # mount the X11 socket
#    -e DISPLAY=unix${DISPLAY} \  # pass the display
