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

# Install Keras
RUN pip install keras

# Install training app
COPY breakout /repositories/breakout

EXPOSE 6006

WORKDIR /repositories/breakout
ENTRYPOINT ["python3"]
CMD [ "play.py" ]
