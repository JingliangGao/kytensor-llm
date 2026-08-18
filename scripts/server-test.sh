#!/bin/bash

# set some variables
CUR_DIR=$(pwd)


# define some functions
inferFun(){
    echo "[INFO] start inference... "
    no_proxy="*" curl -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen_Qwen3-30B-A3B-IQ2_M",
      "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": "Hello"
        }
      ],
      "max_tokens": 8192,
      "temperature": 0.7,
      "stream": false
    }'
    echo "[INFO] inference done."

}

# load model function
loadModelFun(){
    echo "[INFO] load model... "
    curl -s -X POST http://127.0.0.1:8000/model/load
    echo "[INFO] model loaded."
}

# unload model function
unloadModelFun(){
    echo "[INFO] unload model... "
    curl -s -X POST http://127.0.0.1:8000/model/unload
    echo "[INFO] model unloaded."
}

# S3 function
sleepFun(){
    echo "[INFO] sleep... "
    sudo rtcwake -u -s 10 -m mem
    echo "[INFO] sleep done."
}

# S4 function
hibernateFun(){
    echo "[INFO] hibernate... "
    sudo rtcwake -u -s 10 -m disk
    echo "[INFO] hibernate done."
}


# Case 1：推理 → S4 → 恢复
testCase01(){
    inferFun
    hibernateFun
}

# run case 1
echo "[INFO] run case 1... "
testCase01
