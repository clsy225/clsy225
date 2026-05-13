FROM arm64v8/ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-pip \
    python3-serial \
    python3-opencv \
    v4l-utils \
    usbutils \
    git \
    nano \
    vim \
    tmux \
    ros-humble-sensor-msgs \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-usb-cam \
    ros-humble-slam-toolbox \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/ws/src/app
WORKDIR /root/ws

RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo 'if [ -f /root/ws/install/setup.bash ]; then source /root/ws/install/setup.bash; fi' >> /root/.bashrc
