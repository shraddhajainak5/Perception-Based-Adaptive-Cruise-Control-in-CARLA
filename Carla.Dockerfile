FROM carlasim/carla:0.9.15 as carla

USER root 
RUN apt-key del 7fa2af80 \
    && rm /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/cuda.list
RUN apt-get install -y wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update \
    && apt-get install -y curl libomp5 xdg-user-dirs

# We want to make sure Town06 is imported and ready to use for CARLA.
USER carla
WORKDIR /home/carla

RUN curl -sSL https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.15.tar.gz \
  | tar --keep-newer-files -xvz


RUN SDL_VIDEODRIVER=""
CMD unset SDL_VIDEODRIVER; bash ./CarlaUE4.sh -vulkan -RenderOffScreen -nosound
