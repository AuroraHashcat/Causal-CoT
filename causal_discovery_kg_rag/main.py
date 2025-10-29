import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from experiment_logger import ExperimentLogger
from utils import extract_premise, extract_hypothesis
from pipeline.pipeline import CausalDiscoveryPipeline, BatchCasualDiscoveryPipeline
from pipeline.stages import UndirectedSkeletonStage, VStructuresStage, MeekRulesStage, HypothesisEvaluationStage
from llm_client import OpenAIClient, BaseLLMClient, HuggingFaceClient, DeepSeekClient

LOGS_DIR: Path = Path("logs")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Causal Discovery Pipeline with configurable backend and mode."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the split csv file or HuggingFace dataset name (e.g., 'causal-nlp/corr2cause')",
        default="../data/test_dataset.csv"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        help="Dataset split to use when loading from HuggingFace (e.g., 'train', 'test', 'validation')",
        default="test"
    )
    parser.add_argument(
        "--use_huggingface",
        action="store_true",
        help="Load dataset from HuggingFace instead of local CSV file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openai", "huggingface", "deepseek", "deepinfra", "wokka"],
        default="openai",
        help="Choose the LLM backend.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sequential", "batched"],
        default="sequential",
        help="Run pipeline in sequential or batched mode.",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=1,
        help="Number of experiments to run. If greater than dataset length, the whole test set will be used.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for batch processing.",
    )
    return parser.parse_args()


def load_dataset_from_source(args: argparse.Namespace) -> pd.DataFrame:
    """Load dataset from either local CSV or HuggingFace."""
    if args.use_huggingface:
        logging.info(f"Loading dataset from HuggingFace: {args.input_file}, split: {args.dataset_split}")
        
        # 尝试使用不同的镜像站点
        mirrors = [
            None,  # 默认
            "https://hf-mirror.com",  # 国内镜像
        ]
        
        dataset = None
        for mirror in mirrors:
            try:
                if mirror:
                    logging.info(f"尝试使用镜像站点: {mirror}")
                    import os
                    os.environ['HF_ENDPOINT'] = mirror
                else:
                    logging.info("使用默认HuggingFace站点")
                    if 'HF_ENDPOINT' in os.environ:
                        del os.environ['HF_ENDPOINT']
                
                # Load dataset from HuggingFace
                dataset = load_dataset(args.input_file, split=args.dataset_split)
                logging.info(f"成功从 {mirror or '默认站点'} 加载数据集")
                break
                
            except Exception as e:
                logging.warning(f"从 {mirror or '默认站点'} 加载失败: {e}")
                continue
        
        if dataset is None:
            raise Exception("所有HuggingFace镜像站点都无法连接")
            
        try:
            df = dataset.to_pandas()
            logging.info(f"Loaded HuggingFace dataset with {len(df)} rows.")
            
            # Check required columns and adapt if needed
            required_columns = ['input', 'label']
            available_columns = df.columns.tolist()
            logging.info(f"Available columns: {available_columns}")
            
            # Try to map common column names to expected format
            column_mapping = {}
            for col in available_columns:
                col_lower = col.lower()
                if 'input' in col_lower or 'text' in col_lower or 'premise' in col_lower:
                    column_mapping['input'] = col
                elif 'label' in col_lower or 'target' in col_lower or 'answer' in col_lower:
                    column_mapping['label'] = col
                elif 'variable' in col_lower and 'num' in col_lower:
                    column_mapping['num_variables'] = col
                elif 'template' in col_lower:
                    column_mapping['template'] = col
            
            # Rename columns if mapping found
            if column_mapping:
                df = df.rename(columns=column_mapping)
                logging.info(f"Column mapping applied: {column_mapping}")
            
            # Add missing columns with default values if needed
            if 'num_variables' not in df.columns:
                df['num_variables'] = 2  # Default value
            if 'template' not in df.columns:
                df['template'] = 'unknown'  # Default value
            if 'expected_answer' not in df.columns:
                df['expected_answer'] = ''  # Default value
                
        except Exception as e:
            logging.error(f"Error loading HuggingFace dataset: {e}")
            raise
    else:
        # Load from local CSV file
        csv_file = args.input_file
        df = pd.read_csv(csv_file)
        logging.info(f"Loaded dataset from {csv_file} with {len(df)} rows.")
    
    return df


def prepare_input_samples(df: pd.DataFrame, num_experiments: int) -> list[dict]:
    num_experiments = min(num_experiments, len(df))
    sampled_df = df.sample(n=num_experiments, replace=False)

    input_samples = []
    for idx, row in sampled_df.iterrows():
        input_text = row["input"]
        
        # Try to extract premise and hypothesis from input text
        try:
            premise = extract_premise(input_text)
            hypothesis = extract_hypothesis(input_text)
        except Exception as e:
            logging.warning(f"Could not extract premise/hypothesis from sample {idx}: {e}")
            # Fallback: use entire input as premise and empty hypothesis
            premise = input_text
            hypothesis = ""
        
        sample = {
            "sample_id": idx,
            "sample_input": input_text,
            "sample_label": row["label"],
            "sample_num_variables": row.get("num_variables", 2),
            "sample_template": row.get("template", "unknown"),
            "premise": premise,
            "hypothesis": hypothesis
        }
        input_samples.append(sample)
    logging.info(f"Prepared {len(input_samples)} input samples for the pipeline.")
    return input_samples


def create_client(backend: str, batch_size: int) -> BaseLLMClient:
    if backend == "openai":
        client = OpenAIClient(model_id="o3-mini", concurrency=batch_size)
    elif backend == "huggingface":
        client = HuggingFaceClient(max_new_tokens=8192,  batch_size=batch_size, model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
    elif backend == "deepseek":
        client = DeepSeekClient(concurrency=batch_size)
    elif backend == "deepinfra":
        # Deepinfra API - uses OpenAI-compatible interface with Qwen 2.5-72B
        client = OpenAIClient(
            model_id="Qwen/Qwen2.5-72B-Instruct",  # Deepinfra上的Qwen 2.5-72B模型
            concurrency=batch_size,
            base_url="https://api.deepinfra.com/v1/openai",
            api_key_env="DEEPINFRA_API_KEY"
        )
    elif backend == "wokka":
        # Wokka API - 假设也是OpenAI兼容的
        client = OpenAIClient(
            model_id="gpt-4",  # 或Wokka支持的模型名称
            concurrency=batch_size,
            base_url="https://api.wokka.ai/v1"  # 请确认正确的base_url
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    logging.info(f"Using {backend} backend for the pipeline.")
    return client


def post_process_logs(log_file: str) -> None:
    """
    Read the log CSV file, compute confusion matrix and performance metrics,
    then print them out.
    Assumes that each result dictionary contains:
      - "hypothesis_label": a dict with key "hypothesis_answer" (the model's prediction, boolean)
      - "sample_label": the ground truth label (boolean or 'yes'/'no')
    """
    df = pd.read_csv(log_file)
    
    # Convert hypothesis_label to int
    df["hypothesis_label"] = df["hypothesis_label"].astype(int)
    
    # Convert sample_label to int, handling 'yes'/'no' strings
    if df["sample_label"].dtype == 'object':  # String type
        # Handle 'yes'/'no' or other string formats
        df["sample_label"] = df["sample_label"].map(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0)
    else:
        df["sample_label"] = df["sample_label"].astype(int)

    tp = ((df["hypothesis_label"] == 1) & (df["sample_label"] == 1)).sum()
    tn = ((df["hypothesis_label"] == 0) & (df["sample_label"] == 0)).sum()
    fp = ((df["hypothesis_label"] == 1) & (df["sample_label"] == 0)).sum()
    fn = ((df["hypothesis_label"] == 0) & (df["sample_label"] == 1)).sum()

    # Calculate metrics.
    total = len(df)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Confusion Matrix ---")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print("\n--- Performance Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print(f"\n--- Dataset Info ---")
    print(f"Total samples processed: {total}")
    print(f"Results saved to: {log_file}")


def main() -> None:
    args = parse_arguments()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load dataset and prepare input samples.
    df = load_dataset_from_source(args)
    input_samples = prepare_input_samples(df, args.num_experiments)

    # Create the LLM client based on backend choice.
    client = create_client(args.backend, args.batch_size)
    # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

    # Prepare the pipeline
    skeleton_stage = UndirectedSkeletonStage(client=client)
    v_structures_stage = VStructuresStage(client=client)
    meek_rules_stage = MeekRulesStage(client=client)
    hypothesis_evaluation_stage = HypothesisEvaluationStage(client=client)

    job_id = Path(args.input_file).stem
    logger = ExperimentLogger(LOGS_DIR, job_id)
    pipeline: CausalDiscoveryPipeline = CausalDiscoveryPipeline(
        stages=[skeleton_stage, v_structures_stage, meek_rules_stage, hypothesis_evaluation_stage],
        logger=logger,
    )

    results = []
    failed_ids = []
    start_time = time.time()

    if args.mode == "batched":
        logging.info("Running pipeline in batched mode.")
        batch_pipeline = BatchCasualDiscoveryPipeline(pipeline=pipeline, batch_size=args.batch_size)
        results, failed_ids = batch_pipeline.run_batch(input_samples)
    else:
        logging.info("Running pipeline in sequential mode.")
        for sample in tqdm(input_samples, desc="Processing samples"):
            try:
                result = pipeline.run(sample)
                results.append(result)
            except Exception as e:
                failed_ids.append(sample["sample_id"])
                logging.error(f"Error processing sample {sample['sample_id']}: {e}")

    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    # Run results post-processing.
    post_process_logs(str(logger.log_file))

    if failed_ids:
        logging.info(f"Total failed experiments after max retries: {len(failed_ids)}")
        logging.info(f"Failed sample IDs: {failed_ids}")


if __name__  == "__main__":
    main()
