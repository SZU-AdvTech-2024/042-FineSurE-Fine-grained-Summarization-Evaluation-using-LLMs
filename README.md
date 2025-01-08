### 代码架构

- dataset: json格式的FRANK and REALSumm数据集
- finesure: 运行FineSurE进行文本评估

### 模型部署指令

```bash
CUDA_VISIBLE_DEVICES=4 nohup vllm serve shared_models/qwen2.5-14b-instruct --served-model-name qwen2.5-14b-instruct --port 8066 > ./logs/finesure.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup vllm serve shared_models/qwen2.5-32b-instruct --served-model-name qwen2.5-32b-instruct --port 8066 --max-model-len 13760 > ./logs/finesure.log 2>&1 &

# Mixtral-8x7B-v0.1要设置chat-template 
CUDA_VISIBLE_DEVICES=1,4 nohup python -u -m vllm.entrypoints.openai.api_server --model shared_models/Mixtral-8x7B-v0.1 --served-model-name Mixtral-8x7B-v0.1 --tool-call-parser mistral --chat-template shared_models/Mixtral-8x7B-v0.1/tool_chat_template_mistral_parallel.jinja --port 8066 --tensor-parallel-size 2 > ./logs/finesure.log 2>&1 &

```

### 运行指令

#### 任务1：**事实检查（Fact Checking）**

```bash
python finesure/fact-checking.py [input-path] [output-folder]

# example
python finesure/fact-checking.py dataset/frank/frank-data.json result/fact-checking
nohup python finesure/fact-checking.py dataset/frank/frank-data.json result/fact-checking > ./logs/fact_check.log 2>&1 &
```

#### 任务2：**关键事实对齐（Keyfact Alignment）**

```bash
python finesure/keyfact-alignment.py [input-path] [keyfact-path] [output-folder]

# example
python finesure/keyfact-alignment.py dataset/realsumm/realsumm-data.json dataset/realsumm/human-keyfact-list.json result/keyfact-alignment

nohup python finesure/keyfact-alignment.py dataset/realsumm/realsumm-data.json dataset/realsumm/human-keyfact-list.json result/keyfact-alignment > ./logs/keyfacet_alignment.log 2>&1 &
```

#### 评估

```bash
python reproduce-main-results.py results/frank-result-by-qwen2.5-14b-instruct-w-finesure.json results/realsumm-result-by-qwen2.5-14b-instruct-w-finesure.json

python reproduce-main-results.py result/fact-checking/raw-data-by-qwen2.5-32b-instruct.json result/keyfact-alignment/raw-data-by-qwen2.5-32b-instruct.json

python reproduce-main-results.py result/fact-checking/raw-data-by-qwen2.5-32b-instruct.json result/keyfact-alignment/raw-data-by-qwen2.5-32b-instruct.json

```

