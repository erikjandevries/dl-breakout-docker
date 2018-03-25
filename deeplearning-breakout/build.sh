#!/bin/bash

#set -e

source config.sh

echo "Name: ${docker_image_name}"
echo "Version: ${docker_image_version}"
echo "Docker repository: ${docker_repository}"
echo "Image name: ${docker_image}"

docker build -t ${docker_image} .
# docker push ${image_name}
