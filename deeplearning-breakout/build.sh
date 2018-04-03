#!/bin/bash

#set -e

source config.sh

#echo "Name: ${docker_image_name}"
#echo "Version: ${docker_image_version}"
#echo "Docker repository: ${docker_repository}"
#echo ""
echo "Building Docker image: ${docker_image_id}"

docker build -t ${docker_image_id} .
docker image prune -f

# docker push ${image_name}
