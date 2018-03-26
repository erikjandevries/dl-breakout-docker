FROM gcr.io/tensorflow/tensorflow:latest-gpu-py3

MAINTAINER Erik Jan de Vries (github.com/erikjandevries)

# Upgrade pip
RUN pip install --upgrade pip

# Install Intel driver i915
RUN apt-get update \
    && apt-get install -y \
        libgl1-mesa-glx \
        libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# Install OpenAI Gym
RUN apt-get update \
    && apt-get install -y \
        git \
        cmake \
        python-opengl \
    && mkdir /repositories \
    && git clone https://github.com/openai/gym.git /repositories/gym \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repositories/gym
RUN pip install -e '.[atari]'

# Install training app
COPY ./play-random /repositories/play-random

WORKDIR /repositories/play-random
ENTRYPOINT ["python3", "play-random.py"]
