# tb3p 개발·CI용 이미지: ROS 2 Jazzy + TurtleBot3 언더레이 (tb3p.repos 고정 버전)
FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# 기본 도구
RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3-vcstool python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# 언더레이: tb3p.repos 에 적힌 커밋 그대로 받아서 빌드
WORKDIR /opt/tb3_ws
COPY tb3p.repos .
RUN mkdir src && vcs import src < tb3p.repos \
    && apt-get update && rosdep update \
    && rosdep install --from-paths src --ignore-src -r -y \
    && rm -rf /var/lib/apt/lists/* \
    && source /opt/ros/jazzy/setup.bash \
    && colcon build

# 내 코드(오버레이)를 둘 자리
WORKDIR /ws
RUN echo "source /opt/tb3_ws/install/setup.bash" >> ~/.bashrc
