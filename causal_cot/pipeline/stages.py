import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
import json
import re
import os
from datetime import datetime
import math
import itertools
import random
import sys
from pathlib import Path
import traceback
sys.path.append(str(Path(__file__).parent.parent))
from llm_client import BaseLLMClient
from utils import load_prompts, extract_causal_skeleton_json, extract_v_structures_json, \
    extract_directed_edges_literal_format_json, extract_hypothesis_answer, extract_undirected_edges_literal_format_json, \
    extract_initial_construct_json
from search_client import DuckDuckGoSearchClient


class Stage(ABC):
    """
    Base class for all stages in the pipeline.
    Each subclass needs to implement the `prompt_template` attribute.
    """
    prompts: dict[str, str] = load_prompts()
    prompt_template: str = None
    
    # 🆕 类级别的日志文件路径，所有实例共享
    _shared_log_file = None
    _log_initialized = False

    def __init__(self, client: BaseLLMClient, search_client=None, dataset="default", model="default"):
        self.client = client
        self.search_client = search_client
        self.dataset = dataset
        self.model = model
        if self.prompt_template is None:
            raise ValueError("Subclasses must define a prompt_template.")
        if not Stage._log_initialized:
            Stage._initialize_shared_log(self.dataset, self.model)
        
    @classmethod
    def _initialize_shared_log(cls, dataset, model):
        """初始化共享的日志文件，保存在 logs/new 下，命名为 dataset+model+timestamp.log"""
        try:
            # 这里建议恢复自动生成日志路径的代码
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            logs_dir = project_root / "causal_discovery/logs" / dataset / model
            logs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"LLM_Response_{timestamp}.log"
            cls._shared_log_file = logs_dir / log_name
            with open(cls._shared_log_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n\n")
                f.flush()
                os.fsync(f.fileno())
            cls._log_initialized = True
        except Exception as e:
            print(f"❌ 日志文件初始化失败: {e}")
            cls._shared_log_file = None
            cls._log_initialized = True

    def _log_llm_response(self, stage_name: str, prompt: str, response: str, usage=None, operation: str = "main"):
        """统一的LLM响应日志记录方法 - 实时写入版本"""
        timestamp = datetime.now().isoformat()
        
        # 🆕 检查共享日志文件是否可用
        if Stage._shared_log_file is None:
            print(f"⚠️ 日志文件不可用，跳过记录: {stage_name} - {operation}")
            # 🆕 临时创建单独的日志文件作为应急方案
            try:
                temp_log_file = Path(__file__).parent / f"temp_llm_log_{stage_name}_{operation}_{datetime.now().strftime('%H%M%S')}.txt"
                with open(temp_log_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== {stage_name} - {operation} ===\n")
                    f.write(f"Timestamp: {timestamp}\n\n")
                    f.write("PROMPT:\n")
                    f.write(prompt)
                    f.write("\n\nRESPONSE:\n")
                    f.write(response)
                print(f"📝 应急日志已保存: {temp_log_file}")
            except Exception as e:
                print(f"❌ 连应急日志也失败了: {e}")
            return
        
        # 🆕 实时写入到共享日志文件
        try:
            with open(Stage._shared_log_file, 'a', encoding='utf-8') as f:
                # 写入分隔线和标题
                f.write(f"\n{'='*80}\n")
                f.write(f"[{timestamp}] {stage_name} - {operation.upper()}\n")
                f.write(f"{'='*80}\n")
                
                # 文本长度统计
                f.write(f"Text Lengths: Prompt={len(prompt)} chars, Response={len(response)} chars\n\n")
                
                # Prompt内容
                f.write("PROMPT:\n")
                f.write("-" * 40 + "\n")
                f.write(prompt)
                f.write("\n" + "-" * 40 + "\n\n")
                
                # Response内容
                f.write("RESPONSE:\n")
                f.write("-" * 40 + "\n")
                f.write(response)
                f.write("\n" + "-" * 40 + "\n\n")
                
                # 🆕 立即刷新到磁盘，确保实时可见
                f.flush()
                os.fsync(f.fileno())
            
            # 控制台输出确认
            print(f"📝 已记录: {stage_name} - {operation} ({len(response)} chars)")
                
        except Exception as e:
            print(f"❌ 日志写入失败: {e}")
            logging.error(f"Failed to write LLM log to file: {e}")
        
        # 控制台日志记录（保留原有功能）
        logging.info(f"[{stage_name}] LLM {operation} call completed")
        logging.info(f"[{stage_name}] Text lengths - Prompt: {len(prompt)} chars, Response: {len(response)} chars")
    
    def _parse_query_list(self, response: str) -> List[str]:
        """解析查询列表 - 统一方法"""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # 回退解析方法
            match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if match:
                try:
                    return json.loads('[' + match.group(1) + ']')
                except json.JSONDecodeError:
                    pass
            
            # 最后的回退方案
            lines = [line.strip().strip('"').strip("'") 
                    for line in response.split('\n') if line.strip()]
            return [line for line in lines 
                   if line and not line.startswith('[') and not line.startswith(']')][:4]


class InitialConstructStage(Stage):
    """
    Stage -1: Initial graph construction stage - 创建初始的图结构
    在所有其他阶段之前运行，基于premise和hypothesis建立基础图结构
    """
    def __init__(self, client, search_client=None, prompt_type="causal"):
        # 根据类型选择prompt
        self.prompt_template = Stage.prompts[f"initial_construct_{prompt_type}"]
        super().__init__(client, search_client)
    
    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 1. 验证输入
        if "premise" not in input_data or "hypothesis" not in input_data:
            raise ValueError("InitialConstructStage: Input data must contain premise and hypothesis.")
        
        # 2. 构建prompt
        prompt = self.prompt_template.format(
            premise=input_data["premise"],
            hypothesis=input_data["hypothesis"]
        )
        
        # 3. 发送给LLM进行初始构图
        logging.info("InitialConstructStage: Creating initial graph structure")
        response, usage = self.client.complete(prompt=prompt,temperature=0.7)
        
        # 添加LLM响应日志
        self._log_llm_response("InitialConstructStage", prompt, response, usage, "main")
        
        # 4. 🆕 累积式添加到已有结构
        try:
            structure = extract_initial_construct_json(answer=response)
            
            # 🆕 累积添加nodes
            if "nodes" in structure and structure["nodes"]:
                input_data["nodes"].extend(structure["nodes"])
                logging.info(f"InitialConstructStage: Added {len(structure['nodes'])} nodes, total: {len(input_data['nodes'])}")
            
            # 🆕 累积添加directed_edges
            if "edges" in structure and structure["edges"]:
                input_data["edges"].extend(structure["edges"])
                logging.info(f"InitialConstructStage: Added {len(structure['edges'])} ")
            
            # 设置其他字段
            if "causal_question" in structure:
                input_data["causal_question"] = structure["causal_question"]
        
            logging.info(f"InitialConstructStage: Successfully built initial structure")
            
        except Exception as e:
            logging.error(f"InitialConstructStage: Error extracting structure: {e}")
            logging.debug(f"InitialConstructStage: Problematic response: {response}")
        return input_data


class LLMDAGComplementStage(Stage):
    """
    Stage for LLM to complement and refine the causal DAG.
    """
    def __init__(self, client, search_client=None):
        # 根据类型选择prompt
        self.prompt_template = Stage.prompts["LLM_DAG_complement"]
        super().__init__(client, search_client)
    
    def process(self, input_data: dict[str, Any], retry_times: int = 2) -> dict[str, Any]:
        required_keys = {"premise", "hypothesis", "nodes", "edges"}
        if not required_keys.issubset(input_data):
            return self._create_error_result(input_data, "missing_required_keys", f"Missing keys: {required_keys - set(input_data.keys())}")

        for attempt in range(retry_times):
            try:
                prompt1 = self.prompt_template.format(
                    premise=input_data["premise"],
                    hypothesis=input_data["hypothesis"],
                    nodes=input_data["nodes"],
                    edges=input_data["edges"]
                )
                logging.debug(f"Stage1 解析 (attempt {attempt+1})")
                response, usage = self.client.complete(prompt=prompt1,temperature=0.7)
                self._log_llm_response("LLM_DAG_ComplementStage", prompt1, response, usage, "main")
                step1_output = self.convert_step1_to_step2(response)
                input_data["step1_output"] = step1_output  # 作为下一个stage输入
                return input_data
            except Exception as e:
                logging.warning(f"Stage1解析失败 (attempt {attempt+1}): {e}")
                if attempt == retry_times - 1:
                    return self._create_error_result(input_data, "stage1_parsing_failed", str(e))
                
    def convert_step1_to_step2(self, response: str) -> dict:
        """
        从LLM返回的response中提取prompt2所需字段，返回dict，适配新版嵌套结构。
        """
        import re, json
        match = re.search(r'```json\s*({.*?})\s*```', response, re.DOTALL)
        if not match:
            match = re.search(r'({.*})', response, re.DOTALL)
        if not match:
            raise ValueError("未找到符合要求的大JSON格式输出")
        json_str = match.group(1)
        json_str = re.sub(r'//.*', '', json_str)
        data = json.loads(json_str)
        # 只保留新版prompt2需要的字段
        return {
            "causal_pair": data.get("causal_pair", []),
            "complete_graph": data.get("complete_graph", {}),
            "path_analysis": data.get("path_analysis", {})
        }
    def _create_error_result(self, input_data: dict, error_type: str, error_msg: str) -> dict:
        """创建标准化的错误结果，不中断批量处理"""
        logging.error(f"处理失败: {error_type} - {error_msg}")
        
        input_data["success"] = False
        input_data["error_type"] = error_type
        input_data["error_message"] = error_msg
        input_data["causal_effects"] = {
            "ATE": 0.0,
            "NDE": 0.0,
            "NIE": 0.0,
            "TE": 0.0
        }
        input_data["hypothesis_label"] = False
        input_data["production_info"] = {
            "computation_type": "failed",
            "has_colliders": False,
            "collider_warning": None,
                        "mediation_decomposition_valid": False,
                "te_calculation_method": "failed",
            "processing_success": False,
            "error_handled": True
        }
        
        return input_data

class CausalCaculateStage(Stage):
    def __init__(self, client, search_client=None,threshold=0.1):
        # 根据类型选择prompt
        self.prompt_template = Stage.prompts["LLM_DAG_complement"]
        self.threshold = threshold
        super().__init__(client, search_client)

    def _estimate_probability(self, prompt: str) -> float:
        beta_map = {
            "very unlikely": (1.0, 9.0),
            "unlikely":      (2.0, 5.0),
            "possible":      (1.0, 1.0),
            "likely":        (5.0, 2.0),
            "very likely":   (9.0, 1.0),
        }
        synonyms = {
            "very unlikely": ("very unlikely","highly unlikely","almost impossible","near impossible"),
            "unlikely":      ("unlikely","not likely","improbable"),
            "possible":      ("possible","maybe","uncertain","could be"),
            "likely":        ("likely","probable","more likely than not"),
            "very likely":   ("very likely","highly likely","almost certain","near certain","virtually certain"),
        }

        raw = None
        try:
            resp, usage = self.client.complete(prompt=prompt,temperature=0.3)
            raw = str(resp).strip()
            self._log_llm_response("estimate_probability", prompt, raw, usage, "main")

            # 1) numeric first (percent or decimal)
            m = re.search(r'(-?\d+(?:\.\d+)?)\s*%', raw) or re.search(r'(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)', raw, re.I)
            if m:
                p = float(m.group(1))/100.0 if '%' in m.group(0) else float(m.group(1))
                return p

            # 2) bucket via synonyms → Beta sample
            t = re.sub(r"\([^)]*\)", "", raw.lower())
            for label, keys in synonyms.items():
                if any(k in t for k in keys):
                    a, b = beta_map[label]
                    # return random.betavariate(a, b)
                    return a/(a+b)

            # 未匹配到任何结果 → 记录并回退
            self._log_llm_response(
                "estimate_probability_warn",
                prompt,
                f"Failed to parse probability from response: {raw!r}",
                usage,
                "warn"
            )

        except Exception as e:
            # 出错 → 记录异常并回退
            self._log_llm_response(
                "estimate_probability_error",
                prompt,
                f"Exception during probability estimation: {repr(e)}; raw={raw!r}",
                None,
                "error"
            )

        return 0.5

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
            # ===== Step 2: 变量和路径解析 =====
        try:
            step1_output = input_data.get("step1_output")
            if not step1_output:
                return self._create_error_result(input_data, "missing_step1_output", "No step1_output found in input_data.")
            nodes_list = step1_output.get("complete_graph", {}).get("nodes", [])
            nodes = {n.get("node"): n for n in nodes_list if "node" in n}

            # 解析 X/Y 变量
            x_symbol = next((n["node"] for n in nodes_list if n.get("role") == "X"), None)
            y_symbol = next((n["node"] for n in nodes_list if n.get("role") == "Y"), None)
            if not x_symbol or not y_symbol:
                cp = step1_output.get("causal_pair", [])
                if len(cp) == 2:
                    x_symbol = x_symbol or cp[0]
                    y_symbol = y_symbol or cp[1]

            X_var = nodes.get(x_symbol, {}).get("meaning") if x_symbol in nodes else None
            Y_var = nodes.get(y_symbol, {}).get("meaning") if y_symbol in nodes else None
            X_var = X_var or step1_output.get("X_variable", "the cause variable")
            Y_var = Y_var or step1_output.get("Y_variable", "the effect variable")

            # 干预语义
            do_X_1_desc = f"do {X_var}"
            do_X_0_desc = f"not do {X_var}"

            # 路径分析
            pa = step1_output.get("path_analysis", {}) or {}
            path_desc = pa.get("path_description", []) or []
            confounders = pa.get("confounders", []) or []
            intermediates = pa.get("key_intermediate_nodes", []) or []
            colliders = pa.get("colliders", []) or []  # 添加 collider 处理

            has_mediation = False
            has_backdoor = False
            has_collider = False
            has_direct = False
            
            # 确定计算类型 (提高mediation优先级)
            comp_type = "No-path"
            if path_desc:
                has_mediation = any("→" in p and len(p.split("→")) > 2 for p in path_desc)
                has_backdoor = len(confounders) > 0
                has_collider = len(colliders) > 0
                has_direct = any("→" in p and "←" not in p and len(p.split("→")) == 2 for p in path_desc)
                
                # 按优先级顺序判定类型 (mediation 优先级最高)
                if has_mediation and has_backdoor:
                    comp_type = "Mediation+Backdoor"
                elif has_mediation:
                    comp_type = "Mediation"  # 即使同时存在direct，也优先选择mediation
                elif has_backdoor:
                    comp_type = "Backdoor"
                elif has_direct:
                    comp_type = "Direct"
                elif has_collider:
                    comp_type = "Collider"  # 最后才选择 collider
                # 如果以上都不是，保持 "No-path"

            # 构建变量集合
            def build_node_entry(node_symbol: str):
                ninfo = nodes.get(node_symbol, {})
                meaning = ninfo.get("meaning", node_symbol)
                return {"node": node_symbol, "meaning": meaning, "states": ["1", "0"]}

            Z_nodes = [build_node_entry(z) for z in confounders]
            M_nodes = [build_node_entry(m) for m in intermediates]
            
            # 多中介变量时只取第一个
            if len(M_nodes) > 1:
                logging.info(f"检测到{len(M_nodes)}个中介变量，选择第一个: {M_nodes[0]['node']}")
                M_nodes = [M_nodes[0]]

            # 语义二值化
            def mk_binary_semantics(meaning: str) -> dict[str, str]:
                return {"1": f"do {meaning}", "0": f"not do {meaning}"}

            for z in Z_nodes:
                z["state_semantics"] = mk_binary_semantics(z["meaning"])
            for m in M_nodes:
                m["state_semantics"] = mk_binary_semantics(m["meaning"])

            # ===== Step 3: 概率估计（直接从LLM获取）=====
            Z_combinations = list(itertools.product(*[n["states"] for n in Z_nodes])) if Z_nodes else [()]
            M_combinations = list(itertools.product(*[n["states"] for n in M_nodes])) if M_nodes else [()]

            P_Z = {}
            P_M_given = {}
            P_Y_given = {}

            # 估计 p(Z=z)
            if comp_type in {"Backdoor", "Mediation+Backdoor"} and Z_nodes:
                raw_pz = {}
                for z_vals in Z_combinations:
                    cond_z = ", ".join([z["state_semantics"][v] for z, v in zip(Z_nodes, z_vals)])
                    prompt = (
                        f'What is the prior probability that the context holds: "{cond_z}"? '
                        f'Output a single number in [0,1] or a label among [Very unlikely, Unlikely, Possible, Likely, Very likely].'
                    )
                    pz = self._estimate_probability(prompt)
                    raw_pz[z_vals] = float(max(0.0, min(1.0, pz)))
                
                # 归一化
                s = sum(raw_pz.values())
                if s <= 0:
                    logging.warning("所有先验概率为0，使用均匀分布")
                    for z_vals in Z_combinations:
                        P_Z[z_vals] = round(1.0 / len(Z_combinations), 4)
                else:
                    for z_vals, v in raw_pz.items():
                        P_Z[z_vals] = round(v / s, 4)

            # 估计 p(M=m | X, Z)
            if comp_type in {"Mediation", "Mediation+Backdoor"} and M_nodes:
                first_m = M_nodes[0]
                m_states = first_m["states"]
                
                for x_val, z_vals in itertools.product([1, 0], Z_combinations):
                    dist_raw = {}
                    z_sem = ", ".join([z["state_semantics"][v] for z, v in zip(Z_nodes, z_vals)]) if Z_nodes else "no additional conditions"
                    do_desc = do_X_1_desc if x_val == 1 else do_X_0_desc
                    
                    for m_val in m_states:
                        m_desc = first_m["state_semantics"][m_val]
                        prompt = (
                            f"Given {do_desc}, {z_sem}, what is the probability that {m_desc}? "
                            f"Provide only one of [Very unlikely, Unlikely, Possible, Likely, Very likely] "
                            f"or a single number in [0,1]."
                        )
                        p = self._estimate_probability(prompt)
                        dist_raw[m_val] = float(max(0.0, min(1.0, p)))
                    
                    # 归一化
                    s = sum(dist_raw.values())
                    if s <= 0:
                        logging.warning(f"中介概率归一化失败，使用均匀分布: X={x_val}, Z={z_vals}")
                        for m_val in m_states:
                            P_M_given[(m_val, x_val, z_vals)] = round(1.0 / len(m_states), 4)
                    else:
                        for m_val, v in dist_raw.items():
                            P_M_given[(m_val, x_val, z_vals)] = round(v / s, 4)

            # 估计 p(Y | X, Z, M)
            need_m = comp_type in {"Mediation", "Mediation+Backdoor"}
            need_z = comp_type in {"Backdoor", "Mediation+Backdoor"}

            for x_val in [1, 0]:
                do_desc = do_X_1_desc if x_val == 1 else do_X_0_desc
                
                for z_vals in (Z_combinations if need_z else [()]):
                    for m_vals in (M_combinations if need_m else [()]):
                        cond_parts = []
                        if need_z and z_vals != ():
                            cond_parts += [z["state_semantics"][v] for z, v in zip(Z_nodes, z_vals)]
                        if need_m and m_vals != ():
                            cond_parts += [m["state_semantics"][v] for m, v in zip(M_nodes, m_vals)]
                        
                        cond_desc = ", ".join(cond_parts) if cond_parts else "no additional conditions"
                        
                        prompt = (
                            f'Given {do_desc}, {cond_desc}, what is the probability that "{Y_var}" occurs? '
                            f'Output a single number in [0,1] or a label among [Very unlikely, Unlikely, Possible, Likely, Very likely].'
                        )
                        p = self._estimate_probability(prompt)
                        P_Y_given[(x_val, z_vals, m_vals)] = round(float(max(0.0, min(1.0, p))), 4)

            # ===== Step 4: 因果效应计算 =====
            def expect_y_given_do(x_val: int) -> float:
                if comp_type == "Direct":
                    return P_Y_given[(x_val, (), ())]
                elif comp_type == "Backdoor":
                    ey = 0.0
                    for z_vals in Z_combinations:
                        ey += P_Y_given[(x_val, z_vals, ())] * P_Z[z_vals]
                    return round(ey, 4)
                elif comp_type == "Mediation":
                    first_m = M_nodes[0]
                    ey = 0.0
                    for m_val in first_m["states"]:
                        ey += P_Y_given[(x_val, (), (m_val,))] * P_M_given[(m_val, x_val, ())]
                    return round(ey, 4)
                elif comp_type == "Mediation+Backdoor":
                    first_m = M_nodes[0]
                    ey = 0.0
                    for z_vals in Z_combinations:
                        inner = 0.0
                        for m_val in first_m["states"]:
                            inner += P_Y_given[(x_val, z_vals, (m_val,))] * P_M_given[(m_val, x_val, z_vals)]
                        ey += P_Z[z_vals] * inner
                    return round(ey, 4)
                elif comp_type == "Collider":
                    # Collider: 不调整collider，直接计算边际效应
                    return P_Y_given[(x_val, (), ())]
                else:  # No-path
                    return P_Y_given[(x_val, (), ())]

            EY_do1 = expect_y_given_do(1)
            EY_do0 = expect_y_given_do(0)
            ATE = round(EY_do1 - EY_do0, 4)

            # NDE/NIE 计算
            NDE = NIE = 0.0
            if comp_type == "Mediation":
                first_m = M_nodes[0]
                for m_val in first_m["states"]:
                    NDE += (P_Y_given[(1, (), (m_val,))] - P_Y_given[(0, (), (m_val,))]) * P_M_given[(m_val, 0, ())]
                    NIE += P_Y_given[(0, (), (m_val,))] * (P_M_given[(m_val, 1, ())] - P_M_given[(m_val, 0, ())])
                NDE = round(NDE, 4)
                NIE = round(NIE, 4)
            elif comp_type == "Mediation+Backdoor":
                first_m = M_nodes[0]
                for z_vals in Z_combinations:
                    inner_NDE = inner_NIE = 0.0
                    for m_val in first_m["states"]:
                        inner_NDE += (P_Y_given[(1, z_vals, (m_val,))] - P_Y_given[(0, z_vals, (m_val,))]) * P_M_given[(m_val, 0, z_vals)]
                        inner_NIE += P_Y_given[(0, z_vals, (m_val,))] * (P_M_given[(m_val, 1, z_vals)] - P_M_given[(m_val, 0, z_vals)])
                    NDE += P_Z[z_vals] * inner_NDE
                    NIE += P_Z[z_vals] * inner_NIE
                NDE = round(NDE, 4)
                NIE = round(NIE, 4)

            # TE 计算根据 path 类型区分
            if comp_type in {"Mediation", "Mediation+Backdoor"}:
                TE = round(NDE + NIE, 4)  # 中介情况：TE = NDE + NIE
            else:
                TE = ATE  # 其他情况：TE = ATE (No-path, Direct, Backdoor, Collider)

            # ===== Step 5: 结果组装 =====
            causal_results = {
                "ATE": ATE,
                "NDE": NDE,
                "NIE": NIE,
                "TE": TE
            }

            input_data["success"] = True
            input_data["causal_effects"] = causal_results
            input_data["hypothesis_label"] = abs(causal_results["ATE"]) > self.threshold
            
            # 添加简化的生产信息
            input_data["production_info"] = {
                "computation_type": comp_type,
                "total_probabilities_estimated": len(P_Z) + len(P_M_given) + len(P_Y_given),
                "z_variables": len(Z_nodes),
                "m_variables": len(M_nodes),
                "has_colliders": comp_type == "Collider",
                "collider_warning": "Selection bias possible - interpret with caution" if comp_type == "Collider" else None,
                "mediation_decomposition_valid": abs((NDE + NIE) - ATE) <= 0.01 if comp_type in {"Mediation", "Mediation+Backdoor"} else True,
                "te_calculation_method": "NDE + NIE" if comp_type in {"Mediation", "Mediation+Backdoor"} else "ATE",
                "mediation_priority_used": has_mediation and comp_type in {"Mediation", "Mediation+Backdoor"},
                "processing_success": True
            }

            return input_data

        except Exception as e:
            logging.error(f"生产环境处理失败: {e}\n{traceback.format_exc()}")
            return self._create_error_result(input_data, "processing_error", str(e))

    def _create_error_result(self, input_data: dict, error_type: str, error_msg: str) -> dict:
        """创建标准化的错误结果，不中断批量处理"""
        logging.error(f"处理失败: {error_type} - {error_msg}")
        
        input_data["success"] = False
        input_data["error_type"] = error_type
        input_data["error_message"] = error_msg
        input_data["causal_effects"] = {
            "ATE": 0.0,
            "NDE": 0.0,
            "NIE": 0.0,
            "TE": 0.0
        }
        input_data["hypothesis_label"] = False
        input_data["production_info"] = {
            "computation_type": "failed",
            "has_colliders": False,
            "collider_warning": None,
                        "mediation_decomposition_valid": False,
                "te_calculation_method": "failed",
            "processing_success": False,
            "error_handled": True
        }
        
        return input_data


# class BroadRetrievalStage(Stage):
#     """
#     Stage 0: Perform broad retrieval for general background and context,
#     then enhance the initial graph structure with domain knowledge
#     """
#     prompt_template = Stage.prompts["web_search"]
#     enhance_prompt_template = Stage.prompts["search_results_enhancement"]
    
#     def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
#         # 1. 验证输入
#         required_keys = {"premise", "nodes"}
#         if not required_keys.issubset(input_data):
#             raise ValueError(f"BroadRetrievalStage: Input data must contain: {', '.join(required_keys)}.")

#         # 2. 生成广泛的背景搜索查询
#         broad_queries = self._generate_broad_queries(input_data)
        
#         # 3. 执行搜索
#         search_results = self._execute_searches(broad_queries)
        
#         # 🆕 4. 如果有搜索结果，增强图结构；如果没有，保持原始结构
#         if search_results and any(search_results.values()):
#             logging.info("BroadRetrievalStage: Search results available, enhancing graph structure")
#             input_data = self._enhance_graph_with_search_results(input_data, search_results)
#         else:
#             logging.info("BroadRetrievalStage: No search results, keeping original graph structure")
        
#         logging.info(f"BroadRetrievalStage: Completed with {len(broad_queries)} queries")
#         return input_data

#     def _enhance_graph_with_search_results(self, input_data: dict[str, Any], search_results: dict) -> dict[str, Any]:
#         """🆕 使用搜索结果累积增强图结构"""
#         try:
#             # 获取当前图结构信息用于prompt
#             current_nodes = input_data.get("nodes", [])
#             current_undirected_edges = input_data.get("undirected_edges", [])
#             current_directed_edges = input_data.get("directed_edges", [])
            
#             # 使用 self.prompt_template 构建 prompt
#             enhancement_prompt = self.enhance_prompt_template.format(
#                 premise=input_data.get('premise', ''),
#                 hypothesis=input_data.get('hypothesis', ''),
#                 nodes=current_nodes,
#                 directed_edges=current_directed_edges,
#                 undirected_edges=current_undirected_edges
#             )

#             causal_question = input_data.get("causal_question", "")
#             if causal_question:
#                 enhancement_prompt += f"\n\nPay particular attention to whether the causal question holds true： {causal_question}\n"
            
#             # 发送给LLM进行图增强
#             response, usage = self.client.complete(prompt=enhancement_prompt)
            
#             # 记录图增强过程
#             self._log_llm_response("BroadRetrievalStage", enhancement_prompt, response, usage, "graph_enhancement")
            
#             # 解析增强结果
#             enhanced_structure = self._parse_graph_enhancement_response(response)
            
#             # 🆕 累积式添加新的图元素
#             if enhanced_structure:
#                 added_nodes = 0
#                 added_directed = 0
#                 added_undirected = 0
                
#                 # 🆕 添加新节点（避免重复）
#                 if "nodes" in enhanced_structure and enhanced_structure["nodes"]:
#                     existing_node_ids = {node.get("id") for node in input_data["nodes"]}
#                     new_nodes = [node for node in enhanced_structure["nodes"] 
#                                if node.get("id") not in existing_node_ids]
#                     if new_nodes:
#                         input_data["nodes"].extend(new_nodes)
#                         added_nodes = len(new_nodes)
                
#                 # 🆕 添加新的有向边（避免重复）
#                 if "directed_edges" in enhanced_structure and enhanced_structure["directed_edges"]:
#                     existing_directed = {(edge["from"], edge["to"]) for edge in input_data["directed_edges"]}
#                     new_directed = [edge for edge in enhanced_structure["directed_edges"]
#                                     if (edge["from"], edge["to"]) not in existing_directed]
#                     if new_directed:
#                         input_data["directed_edges"].extend(new_directed)
#                         added_directed = len(new_directed)
                
#                 # 🆕 添加新的无向边（避免重复）
#                 if "undirected_edges" in enhanced_structure and enhanced_structure["undirected_edges"]:
#                     existing_undirected = {tuple(sorted(edge)) for edge in input_data["undirected_edges"]}
#                     new_undirected = [edge for edge in enhanced_structure["undirected_edges"]
#                                       if tuple(sorted(edge)) not in existing_undirected]
#                     if new_undirected:
#                         input_data["undirected_edges"].extend(new_undirected)
#                         added_undirected = len(new_undirected)
                
#                 logging.info(f"BroadRetrievalStage: Enhanced graph - Added {added_nodes} nodes, {added_directed} directed edges, {added_undirected} undirected edges")
#                 logging.info(f"BroadRetrievalStage: Total graph size - {len(input_data['nodes'])} nodes, {len(input_data['directed_edges'])} directed edges, {len(input_data['undirected_edges'])} undirected edges")
#             else:
#                 logging.warning("BroadRetrievalStage: No valid enhancement structure extracted")
        
#         except Exception as e:
#             logging.error(f"BroadRetrievalStage: Graph enhancement failed: {e}")
#             # 🆕 错误时不修改已有图结构
    
#         return input_data

    # def _parse_graph_enhancement_response(self, response: str) -> dict:
    #     """解析图增强响应"""
    #     try:
    #         import json
    #         import re
            
    #         # 查找JSON代码块
    #         json_match = re.search(r'```(?:json)?\s*({\s*.*?}\s*)```', response, re.DOTALL)
    #         if json_match:
    #             json_str = json_match.group(1)
    #             data = json.loads(json_str)
                
    #             result = {}
                
    #             # 提取各个字段
    #             for key in ["nodes", "undirected_edges", "directed_edges", "enhanced_premise", "domain_insights"]:
    #                 if key in data:
    #                     result[key] = data[key]
                
    #             logging.debug(f"BroadRetrievalStage: Successfully parsed enhancement response with {len(result)} fields")
    #             return result
    #         else:
    #             logging.warning("BroadRetrievalStage: No JSON found in enhancement response")
    #             return {}
                
    #     except json.JSONDecodeError as e:
    #         logging.error(f"BroadRetrievalStage: JSON parsing failed: {e}")
    #         return {}
    #     except Exception as e:
    #         logging.error(f"BroadRetrievalStage: Failed to parse graph enhancement response: {e}")
    #         return {}

class BroadRetrievalStage(Stage):
    """
    Stage 0: Perform broad retrieval for general background and context,
    then enhance the initial graph structure with domain knowledge.
    输出标准 OUTPUT FORMAT，解析后存到 input_data['step1_output']。
    """
    prompt_template = Stage.prompts["web_search"]
    enhance_prompt_template = Stage.prompts["search_results_enhancement"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        required_keys = {"premise", "nodes"}
        if not required_keys.issubset(input_data):
            raise ValueError(f"BroadRetrievalStage: Input data must contain: {', '.join(required_keys)}.")

        # 1. 生成广泛的背景搜索查询
        broad_queries = self._generate_broad_queries(input_data)

        # 2. 执行搜索
        search_results = self._execute_searches(broad_queries)

        # 3. 用搜索结果和现有input_data生成增强prompt，要求LLM输出标准OUTPUT FORMAT
        if search_results and any(search_results.values()):
            logging.info("BroadRetrievalStage: Search results available, enhancing graph structure")
            enhancement_prompt = self.enhance_prompt_template.format(
                premise=input_data.get('premise', ''),
                hypothesis=input_data.get('hypothesis', ''),
                nodes=input_data.get("nodes", []),
                edges=input_data.get("edges", []),
                search_results=search_results
            )
            response, usage = self.client.complete(prompt=enhancement_prompt)
            self._log_llm_response("BroadRetrievalStage", enhancement_prompt, response, usage, "graph_enhancement")
            step1_output = self._parse_step1_output(response)
            if step1_output:
                input_data["step1_output"] = step1_output
        else:
            logging.info("BroadRetrievalStage: No search results, skipping enhancement")

        logging.info(f"BroadRetrievalStage: Completed with {len(broad_queries)} queries")
        return input_data
    
    def _generate_broad_queries(self, input_data: dict[str, Any]) -> List[str]:
        """生成精准的初始搜索查询"""
        current_nodes = input_data.get("nodes", [])
        current_edges = input_data.get("edges", [])
        
        # 使用 self.prompt_template 构建 prompt
        search_prompt = self.prompt_template.format(
            premise=input_data.get('premise', ''),
            hypothesis=input_data.get('hypothesis', ''),
            nodes=current_nodes,
            edges=current_edges,
        )

        causal_question = input_data.get("causal_question", "")
        if causal_question:
            search_prompt += f"\n\nPay particular attention to whether the causal question holds true： {causal_question}\n"

        try:
            response, usage = self.client.complete(prompt=search_prompt)
            
            # 添加查询生成日志
            self._log_llm_response("BroadRetrievalStage", search_prompt, response, usage, "query_generation")
            
            queries = self._parse_query_list(response)["queries"]
            
            # 限制为最多3个查询，每个查询不超过4个词
            limited_queries = []
            for query in queries[:3]:  # 最多3个
                words = query.split()
                if len(words) <= 4:  # 每个查询最多4个词
                    limited_queries.append(query)
                else:
                    limited_queries.append(' '.join(words[:4]))  # 截断到4个词
        
            logging.info(f"BroadRetrievalStage: Generated {len(limited_queries)} queries: {limited_queries}")
            return limited_queries
        
        except Exception as e:
            logging.error(f"Failed to generate broad queries: {e}")
            return []  # 如果失败，返回空列表而不是回退查询
        
    def _execute_searches(self, queries: List[str]) -> dict:
        """执行搜索查询"""
        search_results = {}
        
        if not queries:
            return search_results
        
        try:
            # 使用DuckDuckGo搜索客户端
            search_client = DuckDuckGoSearchClient(max_results=3)
            
            for query in queries:
                try:
                    logging.info(f"Executing DuckDuckGo search: {query[:50]}...")
                    results = search_client.search(query)
                    if results:
                        search_results[query] = results
                        logging.info(f"✅ Found {len(results)} results for: {query[:30]}...")
                    else:
                        logging.warning(f"⚠️  No results for: {query[:30]}...")
                        search_results[query] = []
                    
                except Exception as e:
                    logging.error(f"Search failed for query '{query}': {e}")
                    search_results[query] = []
                    
        except Exception as e:
            logging.error(f"Failed to initialize DuckDuckGo search client: {e}")
        
        return search_results
    
    def _parse_step1_output(self, response: str) -> dict:
        """解析LLM输出的标准step1_output结构（OUTPUT FORMAT）"""
        import re, json
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                return {}
        json_str = re.sub(r'//.*', '', json_str)
        try:
            data = json.loads(json_str)
            return data
        except Exception as e:
            logging.error(f"BroadRetrievalStage: Failed to parse step1_output: {e}")
            return {}

class UndirectedSkeletonStage(Stage):
    """
    Stage 1: Refine the undirected skeleton based on initial construction and search results
    """
    prompt_template = Stage.prompts["undirected_skeleton"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 1. Validate inputs
        if "premise" not in input_data:
            raise ValueError("Input data must contain Premise.")

        # 3. 构建增强的prompt
        prompt = self._build_enhanced_prompt(input_data)

        causal_question = input_data.get("causal_question", "")
        if causal_question:
            prompt += f"\n\nPay particular attention to whether the causal question holds true： {causal_question}\n"

        # 4. Send request to LLM
        logging.info("UndirectedSkeletonStage: Sending prompt to LLM.")
        response, usage = self.client.complete(prompt=prompt)

        # 🆕 添加LLM响应日志
        self._log_llm_response("UndirectedSkeletonStage", prompt, response, usage, "main")

        # 5. 🆕 累积式精炼图结构 - 不覆盖，只添加
        try:
            skeleton = extract_causal_skeleton_json(answer=response)
            
            # 🆕 累积添加新节点（避免重复）
            if "nodes" in skeleton and skeleton["nodes"]:
                existing_node_ids = {node.get("id") for node in input_data["nodes"]}
                new_nodes = [node for node in skeleton["nodes"] 
                           if node.get("id") not in existing_node_ids]
                if new_nodes:
                    input_data["nodes"].extend(new_nodes)
                    logging.info(f"UndirectedSkeletonStage: Added {len(new_nodes)} new nodes, total: {len(input_data['nodes'])}")
            
            # 🆕 累积添加新无向边（避免重复）
            if "undirected_edges" in skeleton and skeleton["undirected_edges"]:
                refined_edges = []
                for edge in skeleton["undirected_edges"]:
                    if isinstance(edge, list) and len(edge) == 2:
                        refined_edges.append(edge)  # 直接用list结构
                    elif isinstance(edge, dict) and "source" in edge and "target" in edge:
                        refined_edges.append([edge["source"], edge["target"]])
                existing_undirected = {tuple(sorted(e)) for e in input_data["undirected_edges"]}
                new_edges = [e for e in refined_edges if tuple(sorted(e)) not in existing_undirected]
                if new_edges:
                    input_data["undirected_edges"].extend(new_edges)
                    logging.info(f"UndirectedSkeletonStage: Added {len(new_edges)} new undirected edges, total: {len(input_data['undirected_edges'])}")
            
        except Exception as e:
            logging.error("UndirectedSkeletonStage: Error extracting skeleton: %s", e)
            logging.debug("UndirectedSkeletonStage: Problematic response: %s", response)
            # 🆕 错误时不修改已有结构
            logging.info("UndirectedSkeletonStage: Keeping current graph structure due to extraction failure")

        return input_data

    def _build_enhanced_prompt(self, input_data: dict[str, Any]) -> str:
        """构建包含搜索结果的增强prompt"""
        base_prompt = self.prompt_template.format(premise=input_data["premise"])
        
        # 删除所有initial_*相关代码，现在直接基于现有的nodes和edges
        existing_nodes = input_data.get("nodes", [])
        existing_edges = input_data.get("undirected_edges", [])
        
        if existing_nodes or existing_edges:
            base_prompt += f"\n\nCurrent graph structure to refine:\nNodes: {existing_nodes}\nUndirected_Edges: {existing_edges}"
            base_prompt += "\n\nPlease refine this structure based on careful analysis."
        
        # ✅ 修复：添加广泛搜索结果
        if input_data.get('_broad_search_summary'):
            base_prompt += f"\n\nBackground domain context: {input_data['_broad_search_summary']}"
        
        # 添加针对性搜索上下文
        if hasattr(self, '_current_search_summary') and self._current_search_summary:
            base_prompt += f"\n\nRelevant domain context: {self._current_search_summary}"
        
        return base_prompt

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # 简化批处理实现，逐个处理以支持动态搜索
        logging.info("UndirectedSkeletonStage: Processing batch with %d samples.", len(inputs))
        for i, input_data in enumerate(inputs):
            try:
                self.process(input_data)
            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                input_data["nodes"] = None
                input_data["undirected_edges"] = None
        return inputs

class VStructuresStage(Stage):
    """
    Stage for generating the V-structures out of the causal graph and Premise.
    """
    prompt_template = Stage.prompts["v_structures"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 1. Validate inputs
        required_keys = {"premise", "nodes", "undirected_edges"}
        if not required_keys.issubset(input_data):
            raise ValueError(f"Input data must contain: {', '.join(required_keys)}.")

        # Check for None values from previous stages
        if input_data.get("nodes") is None or input_data.get("undirected_edges") is None:
            logging.warning("VStructuresStage: Previous stage returned None values, skipping processing")
            input_data["v_structures"] = None
            return input_data

        # 3. Build enhanced prompt
        try:
            prompt = self._build_enhanced_prompt(input_data)
            causal_question = input_data.get("causal_question", "")
            if causal_question:
                prompt += f"\n\nPay particular attention to whether the causal question holds true： {causal_question}\n"
        except Exception as e:
            logging.error("VStructuresStage: Error formatting prompt: %s", e)
            input_data["v_structures"] = None
            return input_data

        # 4. Send request to LLM
        logging.info("VStructuresStage: Sending prompt to LLM.")
        response, usage = self.client.complete(prompt=prompt)

        # 🆕 添加LLM响应日志
        self._log_llm_response("VStructuresStage", prompt, response, usage, "main")
            # 🆕 修改为累积添加:
        try:
            v_structures_data = extract_v_structures_json(answer=response)
            
            # 🆕 累积添加v结构（避免重复）
            if isinstance(v_structures_data, list):
                # 如果返回列表，直接添加
                existing_v_structures = {str(v) for v in input_data.get("v_structures", [])}
                new_v_structures = [v for v in v_structures_data if str(v) not in existing_v_structures]
                input_data["v_structures"].extend(new_v_structures)
                logging.info(f"VStructuresStage: Added {len(new_v_structures)} new v-structures, total: {len(input_data['v_structures'])}")
            elif isinstance(v_structures_data, dict) and "v_structures" in v_structures_data:
                # 如果返回字典格式
                new_v_structures = v_structures_data["v_structures"]
                if isinstance(new_v_structures, list):
                    existing_v_structures = {str(v) for v in input_data.get("v_structures", [])}
                    unique_new = [v for v in new_v_structures if str(v) not in existing_v_structures]
                    input_data["v_structures"].extend(unique_new)
                    logging.info(f"VStructuresStage: Added {len(unique_new)} new v-structures, total: {len(input_data['v_structures'])}")
        
        except Exception as e:
            logging.error("VStructuresStage: Error extracting V-structures: %s", e)
            logging.debug("VStructuresStage: Problematic response: %s", response)
            # 🆕 错误时不修改已有v_structures
            logging.info("VStructuresStage: Keeping current v-structures due to extraction failure")

        return input_data

    def _build_enhanced_prompt(self, input_data: dict[str, Any]) -> str:
        """构建包含搜索结果的增强prompt"""
        base_prompt = self.prompt_template.format(
            premise=input_data["premise"],
            nodes=input_data["nodes"],
            edges=input_data["undirected_edges"],
        )
        
        # ✅ 修复：添加广泛搜索结果
        if input_data.get('_broad_search_summary'):
            base_prompt += f"\n\nDomain background context: {input_data['_broad_search_summary']}"
        
        # 添加针对性搜索结果
        if hasattr(self, '_current_search_summary') and self._current_search_summary:
            base_prompt += f"\n\nAdditional context from focused search:\n{self._current_search_summary}"
        
        return base_prompt

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logging.info("VStructuresStage: Processing batch with %d samples.", len(inputs))
        for i, input_data in enumerate(inputs):
            try:
                self.process(input_data)
            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                input_data["v_structures"] = None
        return inputs

class MeekRulesStage(Stage):
    """
    Stage for applying Meek's rules to the V-structures.
    """
    prompt_template = Stage.prompts["meek_rules"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 1. Validate inputs
        required_keys = {"premise", "nodes", "undirected_edges", "v_structures"}
        if not required_keys.issubset(input_data):
            raise ValueError(f"Meek rules stage input data must contain: {', '.join(required_keys)}.")

        # 3. Build enhanced prompt
        prompt = self._build_enhanced_prompt(input_data)

        causal_question = input_data.get("causal_question", "")
        if causal_question:
            prompt += f"\n\nPay particular attention to whether the causal question holds true： {causal_question}\n"

        # 4. Send request to LLM
        logging.info("MeekRulesStage: Sending prompt to LLM.")
        response, usage = self.client.complete(prompt=prompt)

        # 🆕 添加LLM响应日志
        self._log_llm_response("MeekRulesStage", prompt, response, usage, "main")

        # 5. Unpack responses and update token usage
        try:
            directed_edges = extract_directed_edges_literal_format_json(answer=response)
            undirected_edges = extract_undirected_edges_literal_format_json(answer=response)   
                
            # 🆕 Meek规则的特殊处理：需要重新定向边，但要保护已有的有向边
            if directed_edges is not None:
                # 保留原有的有向边，添加新的有向边
                existing_directed = {(edge.get("from"), edge.get("to")) for edge in input_data.get("directed_edges", [])}
                new_directed = [edge for edge in directed_edges
                                if (edge.get("from"), edge.get("to")) not in existing_directed]
                
                input_data["directed_edges"].extend(new_directed)
                logging.info(f"MeekRulesStage: Added {len(new_directed)} new directed edges, total: {len(input_data['directed_edges'])}")
            
            if undirected_edges is not None:
                # 只添加新的无向边，数据结构为 ["Node3", "Node4"]
                existing_undirected = {tuple(sorted(edge)) for edge in input_data.get("undirected_edges", [])}
                new_undirected = [edge for edge in undirected_edges
                                if tuple(sorted(edge)) not in existing_undirected]
                input_data["undirected_edges"].extend(new_undirected)
                logging.info(f"MeekRulesStage: Added {len(new_undirected)} new undirected edges, total: {len(input_data['undirected_edges'])}")
            
        except Exception as e:
            logging.error("Error extracting directed edges: %s", e)
            logging.debug("Problematic response: %s", response)
            input_data["directed_edges"] = None
            input_data["undirected_edges"] = None
        return input_data

    def _build_enhanced_prompt(self, input_data: dict[str, Any]) -> str:
        """构建包含搜索结果的增强prompt"""
        base_prompt = self.prompt_template.format(
            premise=input_data["premise"],
            nodes=input_data["nodes"],
            edges=input_data["undirected_edges"],
            v_structures=input_data["v_structures"]
        )
        
        # ✅ 修复：添加广泛搜索结果
        if input_data.get('_broad_search_summary'):
            base_prompt += f"\n\nDomain background context: {input_data['_broad_search_summary']}"
        
        # 添加针对性搜索结果（修复key名称）
        if hasattr(self, '_current_search_summary') and self._current_search_summary:
            base_prompt += f"\n\nAdditional context for edge direction determination:\n{self._current_search_summary}"
        
        return base_prompt

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logging.info("MeekRulesStage: Processing batch with %d samples.", len(inputs))
        for i, input_data in enumerate(inputs):
            try:
                self.process(input_data)
            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                input_data["directed_edges"] = None
                input_data["undirected_edges"] = None
        return inputs

class HypothesisEvaluationStage(Stage):
    """
    Stage for evaluating the hypothesis based on the directed edges.
    """
    prompt_template = Stage.prompts["hypothesis_evaluation"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 1. Validate inputs
        required_keys = {"premise", "nodes", "directed_edges", "hypothesis", "undirected_edges"}
        if not required_keys.issubset(input_data):
            raise ValueError(f"Hypothesis evaluation stage input data must contain: {', '.join(required_keys)}.")


        # 3. Build enhanced prompt (不包含针对性搜索结果)
        prompt = self._build_enhanced_prompt(input_data)

        # 4. Send request to LLM
        logging.info("HypothesisEvaluationStage: Sending prompt to LLM.")
        response, usage = self.client.complete(prompt=prompt)

        # 🆕 添加LLM响应日志
        self._log_llm_response("HypothesisEvaluationStage", prompt, response, usage, "main")

        # 5. Unpack responses and update token usage
        try:
            hypothesis_label = extract_hypothesis_answer(answer=response)
            input_data["hypothesis_label"] = hypothesis_label
            
            logging.info(f"HypothesisEvaluationStage: Hypothesis evaluation result: {hypothesis_label}")
        except Exception as e:
            logging.error("Error extracting hypothesis_label: %s", e)
            logging.debug("Problematic response: %s", response)
            input_data["hypothesis_label"] = 0
        return input_data

    def _build_enhanced_prompt(self, input_data: dict[str, Any]) -> str:
        """构建prompt，只包含广泛搜索结果，不包含针对性搜索"""
        base_prompt = self.prompt_template.format(
            premise=input_data["premise"],
            nodes=input_data["nodes"],
            directed_edges=input_data["directed_edges"],
            undirected_edges=input_data["undirected_edges"],
            hypothesis=input_data["hypothesis"]
        )
        
        # # ✅ 保留：只添加广泛搜索结果（来自BroadRetrievalStage）
        # if input_data.get('_broad_search_summary'):
        #     base_prompt += f"\n\nDomain background context: {input_data['_broad_search_summary']}"
        
        # 🆕 删除：不再添加针对性搜索结果
        # if hasattr(self, '_current_search_summary') and self._current_search_summary:
        #     base_prompt += f"\n\nAdditional context for hypothesis evaluation:\n{self._current_search_summary}"
        
        return base_prompt

    def process_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logging.info("HypothesisEvaluationStage: Processing batch with %d samples.", len(inputs))
        for i, input_data in enumerate(inputs):
            try:
                self.process(input_data)
            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                input_data["hypothesis_label"] = 0
        return inputs


class KnowledgeGraphRetrievalStage(Stage):
    prompt_template = Stage.prompts["kg_search_queries"]

    def __init__(self, client, kg_client):
        super().__init__(client)
        self.kg_client = kg_client  # 传入WikidataClient实例

    def process(self, input_data: dict) -> dict:
        # 1. 生成实体对查询（基于当前因果图节点）
        prompt = self.prompt_template.format(
            premise=input_data["premise"],
            hypothesis=input_data["hypothesis"],
            nodes=input_data.get("nodes", []),
            edges=input_data.get("edges", []),
        )
        response, _ = self.client.complete(prompt=prompt)
        entity_pairs = self._parse_entity_pairs(response)  # 解析实体对

        # 2. 执行知识图谱查询（查找中介和共同原因）
        kg_results = []
        for entity1, entity2 in entity_pairs:
            mediators = self.kg_client.find_mediators(entity1, entity2)
            common_causes = self.kg_client.find_common_causes(entity1, entity2)
            kg_results.append({
                "entity_pair": (entity1, entity2),
                "mediators": mediators,
                "common_causes": common_causes
            })
        input_data["kg_results"] = kg_results

        # 3. 用结果增强因果图，要求LLM输出标准OUTPUT FORMAT
        enhance_prompt = Stage.prompts["kg_search_enhancement"].format(
            nodes=input_data["nodes"],
            edges=input_data["edges"],
            kg_results=kg_results
        )
        enhance_response, _ = self.client.complete(prompt=enhance_prompt)
        # 解析OUTPUT FORMAT，直接放入step1_output
        step1_output = self._parse_step1_output(enhance_response)
        if step1_output:
            input_data["step1_output"] = step1_output

        return input_data

    def _parse_entity_pairs(self, response: str) -> list:
        """解析LLM生成的实体对"""
        try:
            data = json.loads(response.replace("```json", "").replace("```", ""))
            return [tuple(pair) for pair in data.get("entity_pairs", [])]
        except json.JSONDecodeError:
            return []

    def _parse_step1_output(self, response: str) -> dict:
        """解析LLM输出的标准step1_output结构（OUTPUT FORMAT）"""
        import re, json
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                return {}
        json_str = re.sub(r'//.*', '', json_str)
        try:
            data = json.loads(json_str)
            return data
        except Exception as e:
            logging.error(f"KnowledgeGraphRetrievalStage: Failed to parse step1_output: {e}")
            return {}

class RAGEnhancementStage(Stage):
    """基于RAG的因果图增强阶段"""
    prompt_template = Stage.prompts["rag_enhancement"]

    def __init__(self, client, rag_client):
        super().__init__(client)
        self.rag_client = rag_client

    def process(self, input_data: dict) -> dict:
        # 1. 生成RAG查询（基于当前因果图节点）
        query = self._generate_rag_query(input_data)
        if not query:
            return input_data

        # 2. 执行混合搜索
        rag_results = self.rag_client.hybrid_search(query)
        input_data["rag_results"] = rag_results

        # 3. 用RAG结果增强因果图
        step1_output = None
        if rag_results["web_contexts"] or rag_results["kg_relations"]:
            step1_output = self._enhance_graph_and_parse(input_data, rag_results)
        if not step1_output:
            # 即使没有结果，也要写入一个空结构，避免后续报错
            step1_output = {"causal_pair": [], "complete_graph": {}, "path_analysis": {}}
        input_data["step1_output"] = step1_output
        return input_data

    def _generate_rag_query(self, input_data: dict) -> str:
        """生成RAG搜索查询"""
        nodes = input_data.get("nodes", [])
        edges = input_data.get("edges", [])

        prompt = f"""生成一个搜索查询，用于获取以下因果关系的背景信息：
        节点: {nodes}
        关系: {edges}
        前提: {input_data.get('premise')}
        假设: {input_data.get('hypothesis')}
        输出简洁的查询语句（不超过20字）"""

        response, _ = self.client.complete(prompt=prompt)
        return response.strip()

    def _enhance_graph_and_parse(self, input_data: dict, rag_results: dict) -> dict:
        """使用RAG结果增强因果图，并解析为标准step1_output结构"""
        prompt = self.prompt_template.format(
            current_nodes=input_data.get("nodes", []),
            current_edges=input_data.get("edges", []),
            web_contexts=rag_results["web_contexts"],
            kg_relations=rag_results["kg_relations"]
        )

        response, _ = self.client.complete(prompt=prompt)
        return self._parse_step1_output(response)

    def _parse_step1_output(self, response: str) -> dict:
        """解析RAG增强后的标准step1_output结构"""
        import re, json
        # 查找JSON代码块
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                return {}
        json_str = re.sub(r'//.*', '', json_str)
        try:
            data = json.loads(json_str)
            return data
        except Exception as e:
            logging.error(f"RAGEnhancementStage: Failed to parse step1_output: {e}")
            return {}