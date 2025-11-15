#!/bin/bash

# vLLM Serve 双实例脚本 - 充分利用全部 8 个 GPU
# 启动 2 个独立的 vLLM 实例，每个使用 4 个 GPU
# 适用于注意力头数量为 28 的模型（只能被 4 整除）

CKPT_PATH=$1
MODEL_NAME=$2

# 检查参数
if [ -z "$CKPT_PATH" ] || [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <model_path> <model_name>"
    exit 1
fi

echo "=========================================="
echo "启动双实例 vLLM 服务"
echo "模型路径: $CKPT_PATH"
echo "模型名称: $MODEL_NAME"
echo "=========================================="

# 实例 1: GPU 0-3, 端口 8000
echo ""
echo "启动实例 1 (GPU 0-3, 端口 8000)..."
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $CKPT_PATH \
  --served-model-name $MODEL_NAME \
  --chat-template qwen25vl_think_template.jina \
  --max-model-len 8192 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.6 \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype bfloat16 \
  --limit-mm-per-prompt image=5,video=5 \
  --mm-processor-kwargs '{"max_pixels": 1048576, "min_pixels": 262144}' &

PID1=$!
echo "实例 1 PID: $PID1"

# 等待几秒让第一个实例启动
sleep 10

# 实例 2: GPU 4-7, 端口 8001
echo ""
echo "启动实例 2 (GPU 4-7, 端口 8001)..."
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve $CKPT_PATH \
  --served-model-name $MODEL_NAME \
  --chat-template qwen25vl_think_template.jina \
  --max-model-len 8192 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.6 \
  --port 8001 \
  --host 0.0.0.0 \
  --dtype bfloat16 \
  --limit-mm-per-prompt image=5,video=5 \
  --mm-processor-kwargs '{"max_pixels": 1048576, "min_pixels": 262144}' &

PID2=$!
echo "实例 2 PID: $PID2"

echo ""
echo "=========================================="
echo "两个实例已启动！"
echo "实例 1: http://localhost:8000 (GPU 0-3)"
echo "实例 2: http://localhost:8001 (GPU 4-7)"
echo "=========================================="
echo ""
echo "使用 Ctrl+C 停止所有实例"
echo ""

# 等待用户中断
trap "echo '正在停止所有实例...'; kill $PID1 $PID2; exit" INT TERM

# 保持脚本运行
wait

