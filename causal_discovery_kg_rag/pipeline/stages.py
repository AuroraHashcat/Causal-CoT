import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, List
import json
import re
import os
from datetime import datetime

import sys
from pathlib import Path
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

    def __init__(self, client: BaseLLMClient, search_client=None):
        self.client = client
        self.search_client = search_client
        if self.prompt_template is None:
            raise ValueError("Subclasses must define a prompt_template.")
        
        # 🆕 初始化共享日志文件（只初始化一次）- 修复调用方式
        if not Stage._log_initialized:
            Stage._initialize_shared_log()  # 使用类名调用，不是self

    @classmethod
    def _initialize_shared_log(cls):
        """初始化共享的日志文件"""
        try:
            # 🆕 修复：使用绝对路径和更详细的调试信息
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            logs_dir = project_root / "logs"
            
            print(f"🔧 日志目录路径: {logs_dir}")
            
            # 创建日志目录
            logs_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ 日志目录创建成功: {logs_dir}")
            
            # 创建带时间戳的日志文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cls._shared_log_file = logs_dir / f"llm_responses_{timestamp}.log"
            
            print(f"📝 准备创建日志文件: {cls._shared_log_file}")
            
            # 创建日志文件并写入头部信息
            with open(cls._shared_log_file, 'w', encoding='utf-8') as f:
                f.write(f"LLM Response Log - Started at {timestamp}\n")
                f.write("="*80 + "\n\n")
                f.flush()
                os.fsync(f.fileno())
            
            cls._log_initialized = True
            print(f"✅ 共享LLM日志文件已创建: {cls._shared_log_file}")
            
            # 🆕 验证文件是否确实存在
            if cls._shared_log_file.exists():
                print(f"✅ 日志文件验证成功，文件大小: {cls._shared_log_file.stat().st_size} 字节")
            else:
                print(f"❌ 日志文件创建后不存在!")
                cls._shared_log_file = None
            
        except Exception as e:
            print(f"❌ 日志文件初始化失败: {e}")
            import traceback
            print(f"完整错误信息: {traceback.format_exc()}")
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

        # 在每个stage的prompt构建后追加
        causal_question = input_data.get("causal_question", "")
        if causal_question:
            prompt += f"\n\nPay particular attention to whether the causal question holds true： {causal_question}\n"
        
        # 3. 发送给LLM进行初始构图
        logging.info("InitialConstructStage: Creating initial graph structure")
        response, usage = self.client.complete(prompt=prompt)
        
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
            if "directed_edges" in structure and structure["directed_edges"]:
                input_data["directed_edges"].extend(structure["directed_edges"])
                logging.info(f"InitialConstructStage: Added {len(structure['directed_edges'])} directed edges, total: {len(input_data['directed_edges'])}")
            
            # 🆕 累积添加undirected_edges
            if "undirected_edges" in structure and structure["undirected_edges"]:
                input_data["undirected_edges"].extend(structure["undirected_edges"])
                logging.info(f"InitialConstructStage: Added {len(structure['undirected_edges'])} undirected edges, total: {len(input_data['undirected_edges'])}")
            
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
    prompt_template = Stage.prompts["LLM_DAG_complement"]

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 验证输入
        required_keys = {"premise", "hypothesis","nodes", "directed_edges", "undirected_edges"}
        if not required_keys.issubset(input_data):
            raise ValueError(f"LLM_DAG_ComplementStage: Input data must contain: {', '.join(required_keys)}.")

        # 构建prompt
        prompt = self.prompt_template.format(
            premise=input_data["premise"],
            hypothesis = input_data["hypothesis"],
            nodes=input_data["nodes"],
            directed_edges=input_data["directed_edges"],
            undirected_edges=input_data["undirected_edges"]
        )

        causal_question = input_data.get("causal_question", "")
        if causal_question:
            prompt += f"\n\nPay particular attention to whether the causal question holds true： {causal_question}\n"

        logging.info("LLM_DAG_ComplementStage: Sending prompt to LLM.")
        response, usage = self.client.complete(prompt=prompt)

        # 添加LLM响应日志
        self._log_llm_response("LLM_DAG_ComplementStage", prompt, response, usage, "main")

        # 解析LLM补充后的结构
        try:
            structure = extract_initial_construct_json(answer=response)
            # 累积添加nodes
            if "nodes" in structure and structure["nodes"]:
                input_data["nodes"].extend(structure["nodes"])
                logging.info(f"LLM_DAG_ComplementStage: Added {len(structure['nodes'])} nodes, total: {len(input_data['nodes'])}")
            # 累积添加directed_edges
            if "directed_edges" in structure and structure["directed_edges"]:
                input_data["directed_edges"].extend(structure["directed_edges"])
                logging.info(f"LLM_DAG_ComplementStage: Added {len(structure['directed_edges'])} directed edges, total: {len(input_data['directed_edges'])}")
            # 累积添加undirected_edges
            if "undirected_edges" in structure and structure["undirected_edges"]:
                input_data["undirected_edges"].extend(structure["undirected_edges"])
                logging.info(f"LLM_DAG_ComplementStage: Added {len(structure['undirected_edges'])} undirected edges, total: {len(input_data['undirected_edges'])}")
            logging.info("LLM_DAG_ComplementStage: Successfully complemented DAG structure")
        except Exception as e:
            logging.error(f"LLM_DAG_ComplementStage: Error extracting complemented structure: {e}")
            logging.debug(f"LLM_DAG_ComplementStage: Problematic response: {response}")
        return input_data

class BroadRetrievalStage(Stage):
    """
    Stage 0: Perform broad retrieval for general background and context,
    then enhance the initial graph structure with domain knowledge
    """
    prompt_template = Stage.prompts["web_search"]
    enhance_prompt_template = Stage.prompts["search_results_enhancement"]
    
    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 1. 验证输入
        required_keys = {"premise", "nodes"}
        if not required_keys.issubset(input_data):
            raise ValueError(f"BroadRetrievalStage: Input data must contain: {', '.join(required_keys)}.")

        # 2. 生成广泛的背景搜索查询
        broad_queries = self._generate_broad_queries(input_data)
        
        # 3. 执行搜索
        search_results = self._execute_searches(broad_queries)
        
        # 🆕 4. 如果有搜索结果，增强图结构；如果没有，保持原始结构
        if search_results and any(search_results.values()):
            logging.info("BroadRetrievalStage: Search results available, enhancing graph structure")
            input_data = self._enhance_graph_with_search_results(input_data, search_results)
        else:
            logging.info("BroadRetrievalStage: No search results, keeping original graph structure")
        
        logging.info(f"BroadRetrievalStage: Completed with {len(broad_queries)} queries")
        return input_data

    def _generate_broad_queries(self, input_data: dict[str, Any]) -> List[str]:
        """生成精准的初始搜索查询"""
        current_nodes = input_data.get("nodes", [])
        current_undirected_edges = input_data.get("undirected_edges", [])
        current_directed_edges = input_data.get("directed_edges", [])
        
        # 使用 self.prompt_template 构建 prompt
        search_prompt = self.prompt_template.format(
            premise=input_data.get('premise', ''),
            hypothesis=input_data.get('hypothesis', ''),
            nodes=current_nodes,
            directed_edges=current_directed_edges,
            undirected_edges=current_undirected_edges
        )

        causal_question = input_data.get("causal_question", "")
        if causal_question:
            search_prompt += f"\n\nPay particular attention to whether the causal question holds true： {causal_question}\n"

        try:
            response, usage = self.client.complete(prompt=search_prompt)
            
            # 添加查询生成日志
            self._log_llm_response("BroadRetrievalStage", search_prompt, response, usage, "query_generation")
            
            queries = self._parse_query_list(response)
            
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

    def _enhance_graph_with_search_results(self, input_data: dict[str, Any], search_results: dict) -> dict[str, Any]:
        """🆕 使用搜索结果累积增强图结构"""
        try:
            # 获取当前图结构信息用于prompt
            current_nodes = input_data.get("nodes", [])
            current_undirected_edges = input_data.get("undirected_edges", [])
            current_directed_edges = input_data.get("directed_edges", [])
            
            # 使用 self.prompt_template 构建 prompt
            enhancement_prompt = self.enhance_prompt_template.format(
                premise=input_data.get('premise', ''),
                hypothesis=input_data.get('hypothesis', ''),
                nodes=current_nodes,
                directed_edges=current_directed_edges,
                undirected_edges=current_undirected_edges
            )

            causal_question = input_data.get("causal_question", "")
            if causal_question:
                enhancement_prompt += f"\n\nPay particular attention to whether the causal question holds true： {causal_question}\n"
            
            # 发送给LLM进行图增强
            response, usage = self.client.complete(prompt=enhancement_prompt)
            
            # 记录图增强过程
            self._log_llm_response("BroadRetrievalStage", enhancement_prompt, response, usage, "graph_enhancement")
            
            # 解析增强结果
            enhanced_structure = self._parse_graph_enhancement_response(response)
            
            # 🆕 累积式添加新的图元素
            if enhanced_structure:
                added_nodes = 0
                added_directed = 0
                added_undirected = 0
                
                # 🆕 添加新节点（避免重复）
                if "nodes" in enhanced_structure and enhanced_structure["nodes"]:
                    existing_node_ids = {node.get("id") for node in input_data["nodes"]}
                    new_nodes = [node for node in enhanced_structure["nodes"] 
                               if node.get("id") not in existing_node_ids]
                    if new_nodes:
                        input_data["nodes"].extend(new_nodes)
                        added_nodes = len(new_nodes)
                
                # 🆕 添加新的有向边（避免重复）
                if "directed_edges" in enhanced_structure and enhanced_structure["directed_edges"]:
                    existing_directed = {(edge["from"], edge["to"]) for edge in input_data["directed_edges"]}
                    new_directed = [edge for edge in enhanced_structure["directed_edges"]
                                    if (edge["from"], edge["to"]) not in existing_directed]
                    if new_directed:
                        input_data["directed_edges"].extend(new_directed)
                        added_directed = len(new_directed)
                
                # 🆕 添加新的无向边（避免重复）
                if "undirected_edges" in enhanced_structure and enhanced_structure["undirected_edges"]:
                    existing_undirected = {tuple(sorted(edge)) for edge in input_data["undirected_edges"]}
                    new_undirected = [edge for edge in enhanced_structure["undirected_edges"]
                                      if tuple(sorted(edge)) not in existing_undirected]
                    if new_undirected:
                        input_data["undirected_edges"].extend(new_undirected)
                        added_undirected = len(new_undirected)
                
                logging.info(f"BroadRetrievalStage: Enhanced graph - Added {added_nodes} nodes, {added_directed} directed edges, {added_undirected} undirected edges")
                logging.info(f"BroadRetrievalStage: Total graph size - {len(input_data['nodes'])} nodes, {len(input_data['directed_edges'])} directed edges, {len(input_data['undirected_edges'])} undirected edges")
            else:
                logging.warning("BroadRetrievalStage: No valid enhancement structure extracted")
        
        except Exception as e:
            logging.error(f"BroadRetrievalStage: Graph enhancement failed: {e}")
            # 🆕 错误时不修改已有图结构
    
        return input_data

    def _parse_graph_enhancement_response(self, response: str) -> dict:
        """解析图增强响应"""
        try:
            import json
            import re
            
            # 查找JSON代码块
            json_match = re.search(r'```(?:json)?\s*({\s*.*?}\s*)```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                
                result = {}
                
                # 提取各个字段
                for key in ["nodes", "undirected_edges", "directed_edges", "enhanced_premise", "domain_insights"]:
                    if key in data:
                        result[key] = data[key]
                
                logging.debug(f"BroadRetrievalStage: Successfully parsed enhancement response with {len(result)} fields")
                return result
            else:
                logging.warning("BroadRetrievalStage: No JSON found in enhancement response")
                return {}
                
        except json.JSONDecodeError as e:
            logging.error(f"BroadRetrievalStage: JSON parsing failed: {e}")
            return {}
        except Exception as e:
            logging.error(f"BroadRetrievalStage: Failed to parse graph enhancement response: {e}")
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
            input_data["hypothesis_label"] = None
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
                input_data["hypothesis_label"] = None
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
            directed_edges=input_data.get("directed_edges", []),
            undirected_edges=input_data.get("undirected_edges", [])
        )
        response, _ = self.client.complete(prompt=prompt)
        print("\n\n\nresponse: ", response)
        entity_pairs = self._parse_entity_pairs(response)  # 解析实体对

        # 2. 执行知识图谱查询（查找中介和共同原因）
        kg_results = []
        for entity1, entity2 in entity_pairs:
            mediators = self.kg_client.find_mediators(entity1, entity2)
            common_causes = self.kg_client.find_common_causes(entity1, entity2)
            print("\n\n\nmediators: ", mediators)
            kg_results.append({
                "entity_pair": (entity1, entity2),
                "mediators": mediators,
                "common_causes": common_causes
            })
        input_data["kg_results"] = kg_results

        # 3. 用结果增强因果图
        enhance_prompt = Stage.prompts["kg_search_enhancement"].format(
            nodes=input_data["nodes"],
            directed_edges=input_data["directed_edges"],
            undirected_edges=input_data["undirected_edges"],
            kg_results=kg_results
        )
        enhance_response, _ = self.client.complete(prompt=enhance_prompt)
        enhanced_structure = extract_initial_construct_json(enhance_response)

        # 合并增强后的节点和边
        input_data["nodes"].extend(enhanced_structure.get("nodes", []))
        input_data["directed_edges"].extend(enhanced_structure.get("directed_edges", []))
        return input_data

    def _parse_entity_pairs(self, response: str) -> List[tuple]:
        """解析LLM生成的实体对"""
        try:
            data = json.loads(response.replace("```json", "").replace("```", ""))
            return [tuple(pair) for pair in data.get("entity_pairs", [])]
        except json.JSONDecodeError:
            return []



class RAGEnhancementStage(Stage):
    """基于在线RAG的因果图增强阶段"""
    prompt_template = Stage.prompts["rag_enhancement"]

    def __init__(self, client, rag_client):
        super().__init__(client)
        self.rag_client = rag_client

    def process(self, input_data: dict) -> dict:
        # 1. 生成RAG查询
        query = self._generate_rag_query(input_data)
        if not query:
            return input_data

        # 2. 执行在线RAG搜索
        rag_results = self.rag_client.rag_search(query)
        input_data["rag_results"] = rag_results

        # 3. 用RAG结果增强因果图
        if rag_results["rag_contexts"] or rag_results["causal_relations"]:
            enhanced_graph = self._enhance_graph(input_data, rag_results)
            input_data.update(enhanced_graph)

        return input_data

    def _generate_rag_query(self, input_data: dict) -> str:
        """生成RAG搜索查询"""
        nodes = [node["label"] for node in input_data.get("nodes", [])]
        edges = [f"{e['from']}->{e['to']}" for e in input_data.get("directed_edges", [])]

        prompt = f"""生成一个搜索查询，用于获取以下因果关系的背景信息：
        节点: {nodes}
        关系: {edges}
        前提: {input_data.get('premise')}
        假设: {input_data.get('hypothesis')}
        输出简洁的查询语句（不超过20字）"""

        response, _ = self.client.complete(prompt=prompt)
        return response.strip()

    def _enhance_graph(self, input_data: dict, rag_results: dict) -> dict:
        """使用RAG结果增强因果图"""
        print("\n\n\nPrompt Template: ", self.prompt_template)
        prompt = self.prompt_template.format(
            nodes=input_data.get("nodes", []),
            edges=input_data.get("directed_edges", []),
            rag_contexts=rag_results.get("rag_contexts", []),
            causal_relations=rag_results.get("causal_relations", [])
        )

        response, _ = self.client.complete(prompt=prompt)
        return self._parse_enhanced_graph(response)

    def _parse_enhanced_graph(self, response: str) -> dict:
        """解析增强后的因果图"""
        import re
        import json
        json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                self.logger.error("Failed to parse enhanced graph JSON")
        return {}