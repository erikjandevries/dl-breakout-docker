FROM gcr.io/tensorflow/tensorflow:latest-gpu-py3

MAINTAINER Erik Jan de Vries (github.com/erikjandevries)

# 1. Upgrade pip
RUN pip install --upgrade pip


# 2. Install OpenAI Gym
RUN apt-get update \
    && apt-get install -y \
        git \
        cmake \
    && mkdir /repositories \
    && git clone https://github.com/openai/gym.git /repositories/gym

WORKDIR /repositories/gym
RUN pip install -e '.[atari]'

ENTRYPOINT ["/bin/bash"]
