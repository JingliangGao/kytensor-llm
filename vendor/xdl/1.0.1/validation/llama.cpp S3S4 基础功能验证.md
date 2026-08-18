# llama.cpp S3/S4 基础功能验证

## 1. 测试版本

使用以下 `llama.cpp` 版本进行 S3/S4 基础功能验证：

```text
文件：llama.cpp-b8966-rpp-1.0.1.zip
MD5：4227db4b9262556812d7d2bdac44fa12
```

解压并进入编译目录：

```bash
unzip llama.cpp-b8966-rpp-1.0.1.zip
cd llama.cpp-b8966-rpp-1.0.1
mkdir build && cd build
```

------

## 2. 编译

### 2.1 默认配置

```bash
cmake .. \
  -DGGML_RPP=ON \
  -DGGML_RPP_USE_UBATCH=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGGML_RPP_USE_BF16=ON \
  -DLLAMA_CURL=OFF \
  -DGGML_RPP_USE_DFS=ON \
  -DRPP_INCLUDE_DIR=/usr/include/xpu/rpp \
  -DRPP_DRV_LIBRARY=/usr/lib/xpu/rpp/liburpp.so
```

### 2.2 D3000 兼容配置

```bash
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
```

------

## 3. 基础操作（含 S3/S4 流程）

### 3.1 启动 Server

```bash
./llama-server \
  -m <model_path> \
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
```

------

### 3.2 加载模型

```bash
curl -s -X POST http://127.0.0.1:8000/model/load
```

------

### 3.3 推理

```bash
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
```

------

### 3.4 卸载模型 / Context

```bash
curl -s -X POST http://127.0.0.1:8000/model/unload
```

------

### 3.5 S3 / S4 操作说明

#### S4 休眠

1. 点击“开始休眠”
2. 等待指示灯熄灭
3. 按电源键唤醒
4. 出现“联想开天”后系统恢复
5. 重新加载模型并推理验证

#### S3 睡眠

1. 点击“开始睡眠”
2. 等待指示灯熄灭
3. 按电源键唤醒
4. 出现“联想开天”后系统恢复
5. 重新加载模型并推理验证

------

## 4. S3/S4 测试用例

### Case 1：推理 → S4 → 恢复

```text
启动 Server / 加载模型
→ 推理
→ 卸载模型
→ S4 休眠
→ 唤醒
→ 加载模型
→ 推理
```

------

### Case 2：推理 → S3 → 恢复

```text
启动 Server / 加载模型
→ 推理
→ 卸载模型
→ S3 睡眠
→ 唤醒
→ 加载模型
→ 推理
```

------

### Case 3：无推理 → S4 → 恢复

```text
启动 Server / 加载模型
→ 卸载模型
→ S4 休眠
→ 唤醒
→ 加载模型
→ 推理
```

------

### Case 4：无推理 → S3 → 恢复

```text
启动 Server / 加载模型
→ 卸载模型
→ S3 睡眠
→ 唤醒
→ 加载模型
→ 推理
```

