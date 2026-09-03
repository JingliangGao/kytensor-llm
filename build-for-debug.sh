# Description: Build project for debug mode, do not delete current file!!
# Author: JingliangGao
# Date: 2026-09-01

#!/bin/bash

# set environment variables
CUR_DIR=$(pwd)
BACKEND_NAME=$1
RELEASE_DIR=release
BUILD_LOCAL=ON
BUILD_DEBIAN=OFF
MOVE_FILES=OFF
THREAD_NUM=16



# build project locally
if [ ${BUILD_LOCAL} == "ON" ]; then
  echo "[INFO] Build project locally ... "

  SUPPORTED_BACKENDS=(cpu blas cuda vulkan metal)
  if [ -z "${BACKEND_NAME}" ]; then
    BACKEND_NAME=cpu
  elif [[ " ${SUPPORTED_BACKENDS[*]} " != *" ${BACKEND_NAME} "* ]]; then
    echo "[WARNING] ${BACKEND_NAME} not supported, default to cpu"
    BACKEND_NAME=cpu
  fi
  echo "[INFO] Set backend '${BACKEND_NAME}' ..."
  BUILD_DIR="build_${BACKEND_NAME}"

  # if build directory does not exist, create it
  cd ${CUR_DIR}
  if [ -d ${BUILD_DIR} ]; then
    rm -rf ${BUILD_DIR}
  fi
  mkdir ${BUILD_DIR}
  echo "[INFO] Refresh build directory ${BUILD_DIR} ... "

  # set cmake options
  cd ${CUR_DIR}
  echo "[INFO] Set cmake options for ${BUILD_DIR} ... "
  if [ ${BACKEND_NAME} == "cpu" ]; then
    cmake -B ${BUILD_DIR} -S . \
      -DGGML_CPU=ON \
      -DLLAMA_USE_PROFILER=ON
  elif [ ${BACKEND_NAME} == "blas" ]; then
    cmake -B ${BUILD_DIR} -S . \
      -DGGML_BLAS=ON \
      -DLLAMA_USE_PROFILER=ON
  elif [ ${BACKEND_NAME} == "cuda" ]; then
    cmake -B ${BUILD_DIR} -S . \
      -DGGML_CUDA=ON \
      -DLLAMA_USE_PROFILER=ON
  elif [ ${BACKEND_NAME} == "vulkan" ]; then
    cmake -B ${BUILD_DIR} -S . \
      -DGGML_VULKAN=ON \
      -DLLAMA_USE_PROFILER=ON
  elif [ ${BACKEND_NAME} == "metal" ]; then
    cmake -B ${BUILD_DIR} -S . \
      -DGGML_METAL=ON \
      -DLLAMA_USE_PROFILER=ON
  else
    echo "[ERROR] ${BACKEND_NAME} not supported"
    exit 3
  fi

  # build project
  cd ${BUILD_DIR}
  echo "[INFO] Build project for ${BUILD_DIR} ... "
  make -j${THREAD_NUM}

fi

# build project for debian package
if [ ${BUILD_DEBIAN} == "ON" ]; then
  echo "[INFO] Build project for debian package ... " 

  # # AMD packages
  # sudo apt install -y rocm-cmake7.1.0, rocm-core7.1.0, rocm-device-libs7.1.0, comgr-7.1.0, rocm-llvm-dev7.1.0, hsa-rocr7.1.0, hip-dev7.1.0, rocblas-dev7.1.0, hipblas-dev7.1.0, hsa-rocr-dev7.1.0

  # # NVIDIA packages
  # sudo apt install -y libcublas-dev-12-8, cuda-nvcc-12-8, libnvidia-compute-570

  # # Intel packages
  # sudo apt install -y intel-oneapi-devel-2025.2

  # # 格兰菲 (Glenfly)：
  # sudo apt install -y gf-arise-llama-ponn

  # # 后摩智能 (Houmo.AI)：
  # sudo apt install -y houmo-tcim-runtime-xh2

  # # 芯动力 (XDL Technologies)：
  # sudo apt install -y xdl-rpp-runtime, xdl-rpp-runtime-dev

  # # 阿里巴巴 (Alibaba)：
  # sudo apt install -y tongyi-decrypt

  # # base packages
  # sudo apt install -y cmake libpciaccess-dev glslc debhelper libvulkan-dev spirv-headers build-essential ninja-build pkg-config git

  # if build directory does not exist, create it
  cd ${CUR_DIR}
  DEBIAN_BUILD_DIR=build      # do not rename !!
  if [ -d ${DEBIAN_BUILD_DIR} ]; then
    rm -rf ${DEBIAN_BUILD_DIR}
  fi
  
  dpkg-buildpackage -us -uc -b 

fi

# move files
if [ ${MOVE_FILES} == "ON" ]; then
  # move files
  cd ${CUR_DIR}
  if [ -d ${RELEASE_DIR} ]; then
    rm -rf ${RELEASE_DIR}
  fi
  mkdir ${RELEASE_DIR}

  ARCH=$(uname -m)
  if [[ "$ARCH" == "arm"* ]] || [[ "$ARCH" == "aarch64" ]]; then
      echo "当前机器架构为 ARM"
      mkdir -p ${CUR_DIR}/${RELEASE_DIR}/arm
      mv ${CUR_DIR}/../kytensor-llm*.deb ${CUR_DIR}/${RELEASE_DIR}/arm/
      mv ${CUR_DIR}/../kytensor-llm*.buildinfo ${CUR_DIR}/${RELEASE_DIR}/arm/
      mv ${CUR_DIR}/../kytensor-llm*.changes ${CUR_DIR}/${RELEASE_DIR}/arm/
      echo "[INFO] Move files to ${CUR_DIR}/${RELEASE_DIR}/arm/ ... "

  else
      echo "当前机器架构为 x64 (x86_64)"
      mkdir -p ${CUR_DIR}/${RELEASE_DIR}/x64
      mv ${CUR_DIR}/../kytensor-llm*.deb ${CUR_DIR}/${RELEASE_DIR}/x64/
      mv ${CUR_DIR}/../kytensor-llm*.buildinfo ${CUR_DIR}/${RELEASE_DIR}/x64/
      mv ${CUR_DIR}/../kytensor-llm*.changes ${CUR_DIR}/${RELEASE_DIR}/x64/
      echo "[INFO] Move files to ${CUR_DIR}/${RELEASE_DIR}/x64/ ... "
  fi
fi

echo "[INFO] All done! "


