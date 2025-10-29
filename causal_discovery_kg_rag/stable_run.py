#!/usr/bin/env python3
"""
通用稳定版运行脚本：支持多种数据集
"""

import argparse
import logging
import time
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# 导入必要的模块
import sys
sys.path.append(str(Path(__file__).parent))

from experiment_logger import ExperimentLogger
from utils import extract_premise, extract_hypothesis
from pipeline.pipeline import CausalDiscoveryPipeline
from pipeline.stages import UndirectedSkeletonStage, VStructuresStage, MeekRulesStage, HypothesisEvaluationStage, BroadRetrievalStage, InitialConstructStage,LLMDAGComplementStage
from llm_client import OpenAIClient
from search_client import DuckDuckGoSearchClient
from kg_client import CNKnowledgeGraphClient
from pipeline.stages import KnowledgeGraphRetrievalStage
LOGS_DIR: Path = Path("logs/new")



from rag_client import RAGClient
from pipeline.stages import RAGEnhancementStage
def extract_proofwriter_format(input_text: str):
    """专门处理ProofWriter格式的提取"""
    try:
        # ProofWriter数据是JSON格式
        import json
        import ast
        
        # 尝试解析为字典
        if input_text.startswith("{") and input_text.endswith("}"):
            try:
                data = ast.literal_eval(input_text)
                if 'en' in data:
                    en_text = data['en']
                    # 解析en字段: $answer$ ; $proof$ ; $question$ = [问题] ; $context$ = [上下文]
                    if '$question$' in en_text and '$context$' in en_text:
                        parts = en_text.split('$context$ = ', 1)
                        if len(parts) == 2:
                            # 从第一部分提取问题
                            question_part = parts[0]
                            if '$question$ = ' in question_part:
                                question = question_part.split('$question$ = ')[-1].strip(' ;')
                            else:
                                question = ""
                            
                            # 第二部分是上下文
                            context = parts[1].strip()
                            
                            return context, question
            except:
                pass
        
        # 如果不是JSON格式，尝试直接文本解析
        if '$question$' in input_text and '$context$' in input_text:
            parts = input_text.split('$context$ = ', 1)
            if len(parts) == 2:
                question_part = parts[0]
                if '$question$ = ' in question_part:
                    question = question_part.split('$question$ = ')[-1].strip(' ;')
                else:
                    question = ""
                context = parts[1].strip()
                return context, question
                
    except Exception as e:
        pass
    
    raise ValueError("Not ProofWriter format")

def safe_extract_premise_hypothesis(input_text: str, sample_id):
    """安全地提取premise和hypothesis，针对不同数据集格式优化"""
    try:
        # 先尝试原有的提取方式
        premise = extract_premise(input_text)
        hypothesis = extract_hypothesis(input_text) 
        logging.debug(f"Sample {sample_id}: Successfully extracted using standard method")
        return premise, hypothesis
    except Exception as e:
        logging.warning(f"Sample {sample_id}: Could not extract premise/hypothesis using standard method: {e}")
        
        # 专门处理ProofWriter格式
        try:
            premise, hypothesis = extract_proofwriter_format(input_text)
            if premise or hypothesis:
                logging.info(f"Sample {sample_id}: Successfully extracted using ProofWriter format")
                return premise, hypothesis
        except Exception as pw_e:
            logging.debug(f"Sample {sample_id}: ProofWriter extraction failed: {pw_e}")
        
        # 其他备用策略...
        # (保持你现有的备用策略)
        
        # 最后的备用方案
        logging.warning(f"Sample {sample_id}: All extraction methods failed, using full text as premise")
        return input_text, ""

def clean_result_for_csv(result_data: dict) -> dict:
    """清理结果数据，只保留CSV需要的字段"""
    # 定义CSV需要的字段
    csv_fields = {
        'sample_id', 'premise', 'hypothesis', 'sample_label', 
        'nodes', 'undirected_edges', 'v_structures', 'directed_edges',
        'hypothesis_label', 'enhanced_premise'
    }
    
    # 只保留需要的字段
    cleaned_result = {}
    for field in csv_fields:
        if field in result_data:
            cleaned_result[field] = result_data[field]
        else:
            # 为缺失的字段设置默认值
            cleaned_result[field] = None
    
    return cleaned_result

def safe_process_sample(pipeline, sample, max_retries=2):
    """安全地处理单个样本，包含重试机制"""
    sample_id = sample["sample_id"]
    
    for attempt in range(max_retries + 1):
        # try:
        logging.info(f"Processing sample {sample_id} (attempt {attempt + 1}/{max_retries + 1})")
        result = pipeline.run(sample)

        # 🎯 关键修改：清理结果，移除临时字段
        cleaned_result = clean_result_for_csv(result)

        logging.info(f"✅ Sample {sample_id} processed successfully")
        return cleaned_result, None
            
        # except Exception as e:
        #     error_msg = str(e)
        #     logging.error(f"❌ Sample {sample_id} attempt {attempt + 1} failed: {error_msg}")
        #
        #     if attempt < max_retries:
        #         # 等待一段时间再重试
        #         wait_time = (attempt + 1) * 2  # 2, 4 seconds
        #         logging.info(f"Waiting {wait_time} seconds before retry...")
        #         time.sleep(wait_time)
        #     else:
        #         logging.error(f"❌ Sample {sample_id} failed after {max_retries + 1} attempts")
        #         return None, error_msg

def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description='通用因果推理实验脚本')
    parser.add_argument('--input-file', type=str, required=True,
                       help='输入数据文件路径')
    parser.add_argument('--num-experiments', type=int, default=110,
                       help='实验样本数量 (默认: 110)')
    parser.add_argument('--backend', type=str, default='qwen-72b',
                       choices=['qwen-72b', 'qwen-7b','llama-8b','gpt-3.5','claude-3.5','gemini-1.5-flash','ds-r1','claude-3.7-sonnet','o3-mini','gpt-5'],
                       help='LLM后端 (默认: deepinfra)')
    parser.add_argument('--max-retries', type=int, default=1,
                       help='最大重试次数 (默认: 1)')
    parser.add_argument('--search-max-results', type=int, default=3,
                       help='每个搜索查询的最大结果数 (默认: 3)')
    parser.add_argument('--mode', type=int, default=0,
                       help='0:causal reasoning (默认) 1:cot 2:ws')
    
    args = parser.parse_args()
    
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # 从文件路径提取数据集名称，去掉.csv后缀
    from pathlib import Path
    dataset_name = Path(args.input_file).stem



    print(f"📋 配置:")
    print(f"  数据文件: {args.input_file}")
    print(f"  实验数量: {args.num_experiments}")
    print(f"  后端: {args.backend}")
    print(f"  模式: {args.mode}")
    
    # 加载数据集
    try:
        df = pd.read_csv(args.input_file)
    except Exception as e:
        logging.error(f"❌ Failed to load dataset: {e}")
        print(f"❌ 无法加载数据集: {args.input_file}")
        return
    
    # 准备样本
    num_experiments = min(args.num_experiments, len(df))
    sampled_df = df.sample(n=num_experiments, replace=False, random_state=42)

    input_samples = []
    for idx, row in sampled_df.iterrows():
        premise, hypothesis = safe_extract_premise_hypothesis(row["input"], idx)
        
        # 🆕 创建完整的样本字典，预先初始化所有可能需要的字段
        sample = {
            # 基础信息
            "sample_id": idx,
            "sample_input": row["input"],
            "sample_label": row["label"],
            "sample_num_variables": row.get("num_variables", 2),
            "sample_template": row.get("template", "unknown"),
            "premise": premise,
            "hypothesis": hypothesis,
            
            # 🆕 图结构字段 - 预先初始化为空
            "nodes": [],
            "undirected_edges": [],
            "directed_edges": [],
            "v_structures": [],
            
            # 🆕 处理过程字段 - 预先初始化
            "_broad_search_summary": ""
            }
        input_samples.append(sample)

    logging.info(f"✅ Prepared {len(input_samples)} samples with pre-initialized fields")
    
    # 连接LLM
    try:
        if args.backend == "qwen-72b":
            model_id = "Qwen/Qwen2.5-72B-Instruct"
            base_url = "https://api.deepinfra.com/v1/openai"
            api_key_env = "DEEPINFRA_API_KEY"
        elif args.backend == "llama-8b":
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # ← 改为实际可用的模型名
            base_url = "https://api.deepinfra.com/v1/openai"
            api_key_env = "DEEPINFRA_API_KEY"
        elif args.backend == "qwen-7b":
            model_id = "Qwen/Qwen2.5-7B-Instruct"
            base_url = "https://api.deepinfra.com/v1/openai"
            api_key_env = "DEEPINFRA_API_KEY"
        elif args.backend == "gpt-3.5":
            model_id = "gpt-3.5-turbo"
            base_url = "https://4.0.wokaai.com/v1/"
            api_key_env = "WOKKA_API_KEY"
        elif args.backend == "gpt-5":
            model_id = "gpt-5"
            base_url = "https://4.0.wokaai.com/v1/"
            api_key_env = "WOKKA_API_KEY"
        elif args.backend == "claude-3.5":
            model_id = "claude-3-5"
            base_url = "https://api.deepinfra.com/v1/openai"
            api_key_env = "DEEPINFRA_API_KEY"
        elif args.backend == "claude-3.7-sonnet":
            model_id = "anthropic/claude-3-7-sonnet-latest"
            base_url = "https://api.deepinfra.com/v1/openai"
            api_key_env = "DEEPINFRA_API_KEY"
        elif args.backend == "gemini-1.5-flash":
            model_id = "gemini-1.5-flash"
            base_url = "https://4.0.wokaai.com/v1/"
            api_key_env = "WOKKA_API_KEY"
        elif args.backend == "ds-r1":
            model_id = "deepseek-ai/DeepSeek-R1"
            base_url = "https://api.deepinfra.com/v1/openai"
            api_key_env = "DEEPINFRA_API_KEY"
        elif args.backend == "o3-mini":
            model_id = "o3-mini"
            base_url = "https://4.0.wokaai.com/v1/"
            api_key_env = "WOKKA_API_KEY"
            
        client = OpenAIClient(
            model_id=model_id,
            concurrency=1,
            base_url=base_url,
            api_key_env=api_key_env
        )
        
        logging.info(f"✅ Created {args.backend} client")
    except Exception as e:
        logging.error(f"❌ Failed to create client: {e}")
        return

    # try:
    search_client = None
    if args.mode:
        search_client = DuckDuckGoSearchClient(
            max_results=args.search_max_results
        )
        logging.info("DuckDuckGo已启用")

    initial_construct_stage = InitialConstructStage(client=client,prompt_type=dataset_name.split('_')[0], search_client=search_client)
    LLM_DAG_complement_stage = LLMDAGComplementStage(client=client, search_client=search_client)
    broad_retrieval_stage = BroadRetrievalStage(client=client, search_client=search_client)
    skeleton_stage = UndirectedSkeletonStage(client=client, search_client=search_client)
    v_structures_stage = VStructuresStage(client=client, search_client=search_client)
    meek_rules_stage = MeekRulesStage(client=client, search_client=search_client)
    hypothesis_evaluation_stage = HypothesisEvaluationStage(client=client, search_client=search_client)
    kg_client = CNKnowledgeGraphClient()
    knowledge_graph_stage = KnowledgeGraphRetrievalStage(client=client, kg_client=kg_client)
    rag_client = RAGClient(
        max_search_results=args.search_max_results
    )
    rag_enhancement_stage = RAGEnhancementStage(client=client, rag_client=rag_client)

    if args.mode == 0:
        print("mode0: causal reasoning")
        logging.info("mode0: causal reasoning")
        stages = [skeleton_stage, v_structures_stage, meek_rules_stage, hypothesis_evaluation_stage]

    elif args.mode == 1:
        print("mode1: causal-cot")
        logging.info("mode1: causal-cot")
        stages = [initial_construct_stage, LLM_DAG_complement_stage, skeleton_stage, v_structures_stage, meek_rules_stage, hypothesis_evaluation_stage]

    elif args.mode == 2:
        print("mode2: ws")
        logging.info("mode2: ws")
        stages = [initial_construct_stage, broad_retrieval_stage, skeleton_stage, v_structures_stage, meek_rules_stage, hypothesis_evaluation_stage]

    elif args.mode == 3:  # 知识图谱模式
        print("mode3: kg_search")
        logging.info("mode3: kg_search")
        stages = [
            initial_construct_stage,
            knowledge_graph_stage,  # 使用知识图谱检索
            skeleton_stage,
            v_structures_stage,
            meek_rules_stage,
            hypothesis_evaluation_stage
        ]
    if args.mode == 4:
        print("mode4: rag_enhanced")
        logging.info("mode4: rag_enhanced")
        stages = [
            initial_construct_stage,
            rag_enhancement_stage,  # RAG增强阶段
            skeleton_stage,
            v_structures_stage,
            meek_rules_stage,
            hypothesis_evaluation_stage
        ]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    job_id = f"{dataset_name}_{args.backend}_mode{args.mode}_{timestamp}"

    logger = ExperimentLogger(LOGS_DIR, job_id)

    pipeline = CausalDiscoveryPipeline(
        stages=stages,
        logger=logger,
    )

    # except Exception as e:
    #     logging.error(f"❌ 管道创建失败: {e}")
    #     return
    
    # 处理样本
    results = []
    failed_ids = []
    failed_details = []
    
    start_time = time.time()
    
    print(f"\n🔄 开始处理 {len(input_samples)} 个样本...")
    
    for i, sample in enumerate(tqdm(input_samples, desc="Processing samples")):
        print(f"\n--- Sample {i+1}/{len(input_samples)} (ID: {sample['sample_id']}) ---")
        
        result, error = safe_process_sample(pipeline, sample, max_retries=args.max_retries)
        
        if result is not None:
            results.append(result)
            print(f"✅ Success")
        else:
            failed_ids.append(sample["sample_id"])
            failed_details.append({"sample_id": sample["sample_id"], "error": error})
            print(f"❌ Failed: {error}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 结果统计
    success_count = len(results)
    failure_count = len(failed_ids)
    success_rate = (success_count / len(input_samples)) * 100 if input_samples else 0
    
    print(f"\n📊 处理完成统计:")
    print(f"  总样本数: {len(input_samples)}")
    print(f"  成功处理: {success_count}")
    print(f"  处理失败: {failure_count}")
    print(f"  成功率: {success_rate:.1f}%")
    print(f"  总耗时: {total_time:.1f} 秒")
    print(f"  平均每个样本: {total_time/len(input_samples):.1f} 秒")
    
    if failed_ids:
        print(f"\n❌ 失败的样本 IDs: {failed_ids}")
        for detail in failed_details:
            print(f"  Sample {detail['sample_id']}: {detail['error']}")
    
    # 后处理结果
    if success_count > 0:
        try:
            df_results = pd.read_csv(str(logger.log_file))
            
            # 转换标签
            def convert_label(label):
                if isinstance(label, str):
                    return 1 if label.lower() in ['yes', '1', 'true'] else 0
                return int(label)
            
            df_results["sample_label_binary"] = df_results["sample_label"].apply(convert_label)
            df_results["hypothesis_label_binary"] = df_results["hypothesis_label"].astype(int)
            
            # 计算性能指标
            tp = ((df_results["hypothesis_label_binary"] == 1) & (df_results["sample_label_binary"] == 1)).sum()
            tn = ((df_results["hypothesis_label_binary"] == 0) & (df_results["sample_label_binary"] == 0)).sum()
            fp = ((df_results["hypothesis_label_binary"] == 1) & (df_results["sample_label_binary"] == 0)).sum()
            fn = ((df_results["hypothesis_label_binary"] == 0) & (df_results["sample_label_binary"] == 1)).sum()
            
            total = len(df_results)
            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n📈 性能指标 (基于{success_count}个成功样本):")
            print(f"  True Positives: {tp}")
            print(f"  True Negatives: {tn}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            
            print(f"\n💾 详细结果保存到: {logger.log_file}")
            
            # 保存统计信息
            stats_file = logger.log_file.parent / f"{logger.log_file.stem}_statistics.txt"
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(f"{dataset_name}实验统计报告\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("📊 处理完成统计:\n")
                f.write(f"  数据集: {dataset_name}\n")
                f.write(f"  数据文件: {args.input_file}\n")
                f.write(f"  总样本数: {len(input_samples)}\n")
                f.write(f"  成功处理: {success_count}\n")
                f.write(f"  处理失败: {failure_count}\n")
                f.write(f"  成功率: {success_rate:.1f}%\n")
                f.write(f"  后端: {args.backend}\n")
                f.write(f"  模式: {args.mode}\n")  # 🆕
                f.write(f"  总耗时: {total_time:.1f} 秒\n")
                f.write(f"  平均每个样本: {total_time/len(input_samples):.1f} 秒\n\n")
                
                if failed_ids:
                    f.write("❌ 失败的样本:\n")
                    for detail in failed_details:
                        f.write(f"  Sample {detail['sample_id']}: {detail['error']}\n")
                    f.write("\n")
                
                f.write(f"📈 性能指标 (基于{success_count}个成功样本):\n")
                f.write(f"  True Positives: {tp}\n")
                f.write(f"  True Negatives: {tn}\n")
                f.write(f"  False Positives: {fp}\n")
                f.write(f"  False Negatives: {fn}\n")
                f.write(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)\n")
                f.write(f"  Precision: {precision:.4f}\n")
                f.write(f"  Recall:    {recall:.4f}\n")
                f.write(f"  F1 Score:  {f1:.4f}\n\n")
                
                f.write(f"📄 详细实验数据: {logger.log_file.name}\n")
            
            print(f"📊 统计信息保存到: {stats_file}")
            
        except Exception as e:
            logging.error(f"❌ 后处理失败: {e}")
    
    print(f"\n🎉 {dataset_name}实验完成！")
    print(f"📁 结果文件: {logger.log_file}")

if __name__ == "__main__":
    main()
