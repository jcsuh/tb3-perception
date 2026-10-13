# Kalibr (ROS 1 Noetic, Ubuntu 20.04) 이미지
# - 용도: 카메라 내부 파라미터, 카메라-IMU 외부 파라미터·시간 오프셋 캘리브레이션
# - ethz-asl/kalibr의 Dockerfile_ros1_20_04를 바탕으로, 소스를 고정 커밋으로 받아 빌드
# - 빌드: docker build -t tb3p-kalibr docker/kalibr   (20–40분, CI에서는 빌드하지 않음)
FROM osrf/ros:noetic-desktop-full

ARG KALIBR_COMMIT=1f60227442d25e36365ef5f72cd80b9666d73467
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        git wget autoconf automake nano \
        python3-dev python3-pip python3-scipy python3-matplotlib \
        ipython3 python3-wxgtk4.0 python3-tk python3-igraph python3-pyx \
        libeigen3-dev libboost-all-dev libsuitesparse-dev \
        doxygen libopencv-dev \
        libpoco-dev libtbb-dev libblas-dev liblapack-dev libv4l-dev \
        python3-catkin-tools python3-osrf-pycommon \
    && rm -rf /var/lib/apt/lists/*

ENV WORKSPACE=/catkin_ws
RUN mkdir -p $WORKSPACE/src && cd $WORKSPACE \
    && catkin init \
    && catkin config --extend /opt/ros/noetic \
    && catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release

RUN cd $WORKSPACE/src \
    && git clone https://github.com/ethz-asl/kalibr.git \
    && cd kalibr && git checkout $KALIBR_COMMIT

RUN cd $WORKSPACE && catkin build -j$(nproc)

# 초점 거리 자동 초기화가 실패할 때 수동 입력을 허용 (공식 Dockerfile과 같음)
ENV KALIBR_MANUAL_FOCAL_LENGTH_INIT=1
WORKDIR /data
CMD ["bash"]
