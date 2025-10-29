import os
# 1. 设置镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoModel

# 2. 配置参数
MODEL_NAME = "intfloat/e5-base-v2"
TARGET_DIR = "/data/casual_kg_rag/cache/models/e5-base-v2"

# 3. 下载 Tokenizer 和 Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=TARGET_DIR)
model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=TARGET_DIR)