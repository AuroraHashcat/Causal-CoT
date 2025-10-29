# 安装FlashRAG（如果尚未安装）
# !pip install flashrag

from flashrag import FlashRAG
from flashrag.data import load_dataset

# 初始化FlashRAG
rag = FlashRAG()

# 加载FlashRAG提供的基准数据集（以NQ数据集为例）
# 该数据集包含自然问题（Natural Questions）任务的数据
dataset = load_dataset("nq", split="train")
print(f"加载的数据集包含 {len(dataset)} 个样本")

# 查看数据集中的一个样本结构
sample = dataset[0]
print("\n样本结构示例：")
print(f"问题: {sample['question']}")
print(f"答案: {sample['answers']}")
print(f"相关文档片段: {sample['contexts'][:2]}")  # 只显示前2个文档片段

# 进行RAG检索与生成演示
# 1. 构建检索索引
print("\n正在构建检索索引...")
rag.build_index(dataset["contexts"])  # 使用数据集中的文档片段构建索引

# 2. 选择一个问题进行检索增强生成
question = sample["question"]
print(f"\n处理问题: {question}")

# 3. 检索相关文档
retrieved_docs = rag.retrieve(question, top_k=3)  # 检索最相关的3个文档
print("\n检索到的相关文档:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"文档 {i}: {doc[:200]}...")  # 只显示前200个字符

# 4. 基于检索到的文档生成答案
generated_answer = rag.generate(question, retrieved_docs)
print(f"\n生成的答案: {generated_answer}")

# 5. 评估生成答案与参考答案的相似度
reference_answers = sample["answers"]
score = rag.evaluate(generated_answer, reference_answers)
print(f"\n答案评估分数: {score}")