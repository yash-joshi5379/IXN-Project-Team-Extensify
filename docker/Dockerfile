FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-vision-opencv \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglfw3 \
    libglew-dev \
    x11-apps \
    mesa-utils \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    mujoco \
    "numpy<2" \
    scipy \
    pyarrow \
    pandas \
    "opencv-python-headless<4.10" \
    pynput

RUN mkdir -p /extend_robotics_ws/src

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "source /extend_robotics_ws/install/setup.bash 2>/dev/null || true" >> /root/.bashrc

WORKDIR /extend_robotics_ws

CMD ["/bin/bash"]
