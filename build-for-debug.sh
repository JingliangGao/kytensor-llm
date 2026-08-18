#!/bin/bash

# set some variables
CUR_DIR=$(pwd)
VENDOR_DIR=${CUR_DIR}/vendor/innosilicon/1.0.1               # vendor directory
MODEL_PATH=/home/kylin/gjl/model/Qwen_Qwen3-8B-Q4_K_M.gguf   # model path
CONFIG_FLAG="default"                                        # "default", "d3000" 
BUILD_DIR=build_innosilicon                                  # build directory
RELEASE_DIR=release                                          # release directory
INSTALL_DRIVER=OFF                                           # install driver
INSTALL_RUNTIMETIME=OFF                                      # install runtime 
BUILD_PROJECT=OFF                                            # build project
BUILD_DEBIAN=OFF                                             # build debian 
TEST_SERVER=OFF                                              # test server
MOVE_FILES=OFF                                               # move files 

# install driver
if [ ${INSTALL_DRIVER} == "ON" ]; then
    echo "[INFO] Install driver ... "
    cd ${VENDOR_DIR}/deb/
    sudo dpkg -i xdl-rpp-dkms_1.0.0_all.deb

    echo "[INFO] Check driver installation status."
    echo "******************************************"
    ls -ll /usr/src/rpp-dkms-*/
    ls -ll /lib/modules/$(uname -r)/kernel/extra/
    echo "******************************************"
fi

# install runtime
if [ ${INSTALL_RUNTIMETIME} == "ON" ]; then
    echo "[INFO] Install runtime ... "
    cd ${VENDOR_DIR}/deb/

    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm"* ]] || [[ "$ARCH" == "aarch64" ]]; then
      echo "[INFO] current machine architecture is: ARM"
      sudo dpkg -i xdl-rpp-runtime_1.0.1_arm64.deb
    else
      echo "[INFO] current machine architecture is: x64 (x86_64)"
      sudo dpkg -i xdl-rpp-runtime_1.0.1_x64.deb
    fi
    sudo dpkg -i xdl-rpp-runtime-dev_1.0.0_all.deb
fi

# refresh build directory
if [ ${BUILD_PROJECT} == "ON" ]; then

    echo "[INFO] Refresh build directory ... "
    cd ${CUR_DIR}
    if [ -d ${BUILD_DIR} ]; then
        rm -rf ${BUILD_DIR}
    fi
    mkdir -p ${BUILD_DIR}

    # set cmake config
    echo "[INFO] Set cmake config ... "
    cd ${CUR_DIR}/${BUILD_DIR}
    if [ CONFIG_FLAG == "d3000" ]; then
        echo "[INFO] Set d3000 cmake config ... "
        cmake .. \
          -DGGML_RPP=ON \
          -DGGML_RPP_USE_UBATCH=ON \
          -DCMAKE_BUILD_TYPE=Debug \
          -DGGML_RPP_USE_BF16=ON \
          -DLLAMA_CURL=OFF \
          -DGGML_RPP_USE_DFS=ON \
          -DGGML_CPU_NATIVE=OFF \
          -DGGML_NATIVE=OFF \
          -DGGML_CPU_ARM_ARCH=armv8-a \
          -DGGML_CPU_LLAMAFILE=OFF \
          -DRPP_INCLUDE_DIR=/usr/include/xpu/rpp \
          -DRPP_DRV_LIBRARY=/usr/lib/xpu/rpp/liburpp.so
    else
        echo "[INFO] Set default cmake config ... "
        cmake .. \
          -DGGML_RPP=ON \
          -DGGML_RPP_USE_UBATCH=ON \
          -DCMAKE_BUILD_TYPE=Debug \
          -DGGML_RPP_USE_BF16=ON \
          -DLLAMA_CURL=OFF \
          -DGGML_RPP_USE_DFS=ON \
          -DRPP_INCLUDE_DIR=/usr/include/xpu/rpp \
          -DRPP_DRV_LIBRARY=/usr/lib/xpu/rpp/liburpp.so
    fi

    # build
    echo "[INFO] Build llama.cpp-b8966-rpp-1.0.1 ... "
    cd ${CUR_DIR}/${BUILD_DIR}
    make -j$(nproc)
fi

# refresh build directory
if [ ${BUILD_DEBIAN} == "ON" ]; then
    echo "[INFO] Install build dependencies ... "
    # sudo apt-get update
    # sudo apt-get install -y build-essential cmake ninja-build pkg-config git debhelper-compat

    echo "[INFO] Build debian package ... "
    cd ${CUR_DIR}
    dpkg-buildpackage -uc -us
fi

# test server
if [ ${TEST_SERVER} == "ON" ]; then
    echo "[INFO] Run server ... "
    cd ${CUR_DIR}/${BUILD_DIR}/bin
    ./llama-server \
      -m ${MODEL_PATH} \
      --host 0.0.0.0 \
      --port 8000 \
      -c 4096 \
      -ctk bf16 \
      -ctv bf16 \
      --no-warmup \
      -ub 512 \
      --fit off \
      --jinja \
      --keep 256
fi

# move files
if [ ${MOVE_FILES} == "ON" ]; then
    echo "[INFO] Refresh release directory: ${RELEASE_DIR}"
    cd ${CUR_DIR}
    if [ -d ${RELEASE_DIR} ]; then
      rm -rf ${RELEASE_DIR}
    fi
    mkdir -p ${RELEASE_DIR}

    echo "[INFO] Move files ... "
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm"* ]] || [[ "$ARCH" == "aarch64" ]]; then
      mkdir -p ${RELEASE_DIR}/arm64
      if [ -f ${CUR_DIR}/../kytensor-llm_2.0.0-ok19k1.5_arm64.deb ]; then
        cp -f ${CUR_DIR}/../*.deb ${RELEASE_DIR}/x64/
        cp -f ${CUR_DIR}/../kytensor-llm_2.0.0-ok19k1.5.tar.xz ${RELEASE_DIR}/x64/
        cp -f ${CUR_DIR}/../kytensor-llm_2.0.0-ok19k1.5_arm64.changes ${RELEASE_DIR}/x64/
        cp -f ${CUR_DIR}/../kytensor-llm_2.0.0-ok19k1.5.dsc ${RELEASE_DIR}/x64/
        cp -f ${CUR_DIR}/../kytensor-llm_2.0.0-ok19k1.5_arm64.buildinfo ${RELEASE_DIR}/x64/     
      fi
    else
      mkdir -p ${RELEASE_DIR}/x64
      if [ -f ${CUR_DIR}/../kytensor-llm_2.0.0-ok19k1.5_amd64.deb ]; then
        cp -f ${CUR_DIR}/../*.deb ${RELEASE_DIR}/x64/
        cp -f ${CUR_DIR}/../kytensor-llm_2.0.0-ok19k1.5.tar.xz ${RELEASE_DIR}/x64/
        cp -f ${CUR_DIR}/../kytensor-llm_2.0.0-ok19k1.5_amd64.changes ${RELEASE_DIR}/x64/
        cp -f ${CUR_DIR}/../kytensor-llm_2.0.0-ok19k1.5.dsc ${RELEASE_DIR}/x64/
        cp -f ${CUR_DIR}/../kytensor-llm_2.0.0-ok19k1.5_amd64.buildinfo ${RELEASE_DIR}/x64/        
      fi
    fi

fi


echo "[INFO] All done!"
