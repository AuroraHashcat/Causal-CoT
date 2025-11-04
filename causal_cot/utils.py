import json
import re
from functools import lru_cache
from typing import Any
import logging

import yaml


def chunk_list(lst: list[Any], chunk_size: int) -> list[list[Any]]:
    """
    Splits the list 'lst' into chunks of size 'chunk_size'.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


@lru_cache(maxsize=1)
def load_prompts(file_path: str = "prompts.yaml") -> dict[str, str]:
    """
    Load prompts from a YAML file.

    Parameters:
    - file_path (str): Path to the YAML file containing prompts.

    Returns:
    - dict: A dictionary where keys are prompt names and values are the corresponding prompts.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def extract_premise(text: str) -> str:
    """
    Extracts Premise from the input text.

    Parameters:
    - text (str): The input text containing "Premise:" and possibly "Hypothesis:".

    Returns:
    - str: The extracted premise.
    """
    # Use regex to extract text between "Premise:" and "Hypothesis:"
    match = re.search(r"Premise:\s*(.*?)\s*Hypothesis:", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # If "Hypothesis:" is not present, extract everything after "Premise:"
        match = re.search(r"Premise:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            raise RuntimeError("Failed to extract premise from the text.")


def extract_hypothesis(text: str) -> str:
    """
    Extracts Hypothesis from the input text.

    Parameters:
    - text (str): The input text containing "Hypothesis:".

    Returns:
    - str: The extracted hypothesis.
    """
    # Use regex to extract everything after "Hypothesis:"
    match = re.search(r"Hypothesis:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        raise RuntimeError("Failed to extract hypothesis from the text.")

def fix_json_latex(json_str: str) -> str:
    """
    把所有单反斜杠替换为双反斜杠，避免 LaTeX 公式导致 JSON 解析报错。
    """
    # 只处理未转义的反斜杠
    return re.sub(r'(?<!\\)\\', r'\\\\', json_str)


def extract_causal_skeleton_json(answer: str) -> dict[str, Any]:
    try:
        # 优先找三反引号包裹的 JSON
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            json_str = fix_json_latex(json_str)  # 加这一步
            data = json.loads(json_str)
        else:
            # 兜底：直接找第一个大括号包裹的 JSON
            json_match = re.search(r'{.*}', answer, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                json_str = fix_json_latex(json_str)  # 加这一步
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in the answer.")

        result = {}
        if "nodes" in data:
            result["nodes"] = data["nodes"]
        else:
            raise ValueError("No nodes found in the JSON data.")

        if "undirected_edges" in data:
            # 保证每个边是list且去重
            edges = []
            seen = set()
            for edge in data["undirected_edges"]:
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    sorted_edge = tuple(sorted(edge))
                    if sorted_edge not in seen:
                        edges.append([edge[0], edge[1]])
                        seen.add(sorted_edge)
            result["undirected_edges"] = edges
        else:
            raise ValueError("No undirected_edges found in the JSON data.")

        return result
    except Exception as e:
        raise RuntimeError(f"Failed to extract causal skeleton: {e}")


def extract_v_structures_json(answer: str) -> list[tuple]:
    """
    Extract the v-structures from the provided LLM answer string using the JSON format.

    :param answer: The answer returned by the LLM API.
    :return: A list of v-structures extracted from the answer.
    """
    try:
        # First approach: Find JSON block with triple quotes
        json_match = re.search(r'```(?:json)?\s*({\s*".*?}\s*)```', answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            json_str = fix_json_latex(json_str)  # 加这一步
            data = json.loads(json_str)

            # Extract v-structures
            if "v_structures" in data:
                v_structures = [tuple(v_struct) for v_struct in data["v_structures"]]
                return v_structures

        # Second approach: Find all potential JSON objects and test each one
        json_blocks = re.findall(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', answer)
        for json_str in json_blocks:
            try:
                data = json.loads(json_str)
                if "v_structures" in data:
                    v_structures = [tuple(v_struct) for v_struct in data["v_structures"]]
                    return v_structures
            except json.JSONDecodeError:
                continue

        # If no v_structures found in JSON, look for v-structures in text format
        v_structures_pattern = r'v.?structures?:?\s*\[(.*?)\]'
        v_struct_match = re.search(v_structures_pattern, answer, re.IGNORECASE | re.DOTALL)
        if v_struct_match:
            content = v_struct_match.group(1)
            # Extract arrays like ["A", "B", "C"]
            array_pattern = r'\[\s*"([A-E])"\s*,\s*"([A-E])"\s*,\s*"([A-E])"\s*\]'
            v_structures = []
            for match in re.finditer(array_pattern, content):
                v_structures.append(tuple(match.groups()))
            if v_structures:
                return v_structures

        raise ValueError("No v-structures found in the answer.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract v-structures: {e}")


def extract_directed_edges_literal_format_json(answer: str) -> list:
    """
    Extract the directed edges from the provided LLM answer string using the expected JSON format.
    Always returns [{"from": ..., "to": ...}, ...]
    """
    try:
        # First approach: Find JSON block delimited by triple backticks.
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            json_str = fix_json_latex(json_str)  # 加这一步
            data = json.loads(json_str)
            if "final_graph" in data and "directed_edges" in data["final_graph"]:
                if not data["final_graph"]["directed_edges"]:
                    return []
                directed_edges = []
                for edge in data["final_graph"]["directed_edges"]:
                    if isinstance(edge, dict) and "from" in edge and "to" in edge:
                        directed_edges.append({"from": edge["from"], "to": edge["to"]})
                    elif isinstance(edge, (list, tuple)) and len(edge) == 2:
                        directed_edges.append({"from": edge[0], "to": edge[1]})
                return directed_edges

        # Second approach: Search for any JSON objects within the answer.
        json_blocks = re.findall(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', answer)
        for json_str in json_blocks:
            try:
                data = json.loads(json_str)
                if "final_graph" in data and "directed_edges" in data["final_graph"]:
                    if not data["final_graph"]["directed_edges"]:
                        return []
                    directed_edges = []
                    for edge in data["final_graph"]["directed_edges"]:
                        if isinstance(edge, dict) and "from" in edge and "to" in edge:
                            directed_edges.append({"from": edge["from"], "to": edge["to"]})
                        elif isinstance(edge, (list, tuple)) and len(edge) == 2:
                            directed_edges.append({"from": edge[0], "to": edge[1]})
                    return directed_edges
            except json.JSONDecodeError:
                continue

        # Third approach: Look for directed edges in text format from the final_graph section.
        edges_pattern = r'"directed_edges"\s*:\s*\[(.*?)\]'
        edges_match = re.search(edges_pattern, answer, re.IGNORECASE | re.DOTALL)
        if edges_match:
            content = edges_match.group(1).strip()
            if not content:
                return []
            edge_pattern = r'\{\s*"from"\s*:\s*"([^"]+)"\s*,\s*"to"\s*:\s*"([^"]+)"\s*\}'
            directed_edges = []
            for match in re.finditer(edge_pattern, content):
                directed_edges.append({"from": match.group(1), "to": match.group(2)})  # ✅ 改为dict
            return directed_edges

        raise ValueError("No directed edges found in the answer.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract directed edges: {e}")


def extract_undirected_edges_literal_format_json(answer: str) -> list:
    """
    Extract the undirected edges from the provided LLM answer string using the expected JSON format.

    Expected JSON format:
    {
      "final_graph": {
        "directed_edges": [
          { "from": "Node1", "to": "Node2" },
          { "from": "Node2", "to": "Node3" }
        ],
        "undirected_edges": [
          ["Node3", "Node4"],
          ["Node5", "Node6"]
        ]
      }
    }

    :param answer: The answer returned by the LLM API.
    :return: A list of undirected edges (tuples representing an undirected connection) extracted from the answer.
    """
    try:
        # First approach: Find JSON block delimited by triple backticks.
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            json_str = fix_json_latex(json_str)  # 加这一步
            data = json.loads(json_str)
            if "final_graph" in data and "undirected_edges" in data["final_graph"]:
                if not data["final_graph"]["undirected_edges"]:
                    return []
                undirected_edges = []
                for edge in data["final_graph"]["undirected_edges"]:
                    if isinstance(edge, list) and len(edge) == 2:
                        undirected_edges.append([edge[0], edge[1]])
                    elif isinstance(edge, tuple) and len(edge) == 2:
                        undirected_edges.append([edge[0], edge[1]])
                return undirected_edges

        # Second approach: Search for any JSON objects within the answer.
        json_blocks = re.findall(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', answer)
        for json_str in json_blocks:
            try:
                data = json.loads(json_str)
                if "final_graph" in data and "undirected_edges" in data["final_graph"]:
                    if not data["final_graph"]["undirected_edges"]:
                        return []
                    undirected_edges = []
                    for edge in data["final_graph"]["undirected_edges"]:
                        if isinstance(edge, list) and len(edge) == 2:
                            undirected_edges.append([edge[0], edge[1]])
                        elif isinstance(edge, tuple) and len(edge) == 2:
                            undirected_edges.append([edge[0], edge[1]])
                    return undirected_edges
            except json.JSONDecodeError:
                continue

        # Third approach: Look for undirected_edges in text format from the final_graph section.
        edges_pattern = r'"undirected_edges"\s*:\s*\[(.*?)\]'
        edges_match = re.search(edges_pattern, answer, re.IGNORECASE | re.DOTALL)
        if edges_match:
            content = edges_match.group(1).strip()
            if not content:
                return []
            edge_pattern = r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]'
            undirected_edges = []
            for match in re.finditer(edge_pattern, content):
                undirected_edges.append([match.group(1), match.group(2)])
            return undirected_edges

        raise ValueError("No undirected edges found in the answer.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract undirected edges: {e}")


def extract_hypothesis_answer(answer: str) -> bool:
    """
    Extract the hypothesis answer (True/False) from the provided LLM answer string.

    Parameters:
    - answer (str): The answer returned by the LLM API.

    Returns:
    - bool: The extracted hypothesis answer as a Python boolean.

    Raises:
    - RuntimeError: If extraction fails.
    """
    try:
        # First approach: Try to parse the entire answer as JSON
        try:
            data = json.loads(answer)
            if "hypothesis_answer" in data:
                return bool(data["hypothesis_answer"])
        except json.JSONDecodeError:
            pass

        # Second approach: Find JSON block with triple quotes
        json_match = re.search(r'```(?:json)?\s*(.*?)```', answer, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                json_str = fix_json_latex(json_str)  # 加这一步
                data = json.loads(json_str)
                if "hypothesis_answer" in data:
                    return bool(data["hypothesis_answer"])
            except json.JSONDecodeError:
                pass

        # Third approach: Try to find complete JSON objects in the text
        potential_json_start = 0
        while potential_json_start < len(answer):
            start_idx = answer.find('{', potential_json_start)
            if start_idx == -1:
                break

            # Find matching closing brace by tracking brace balance
            open_braces = 1
            for end_idx in range(start_idx + 1, len(answer)):
                if answer[end_idx] == '{':
                    open_braces += 1
                elif answer[end_idx] == '}':
                    open_braces -= 1

                if open_braces == 0:
                    # Found a potential JSON object
                    try:
                        json_str = answer[start_idx:end_idx + 1]
                        data = json.loads(json_str)
                        if "hypothesis_answer" in data:
                            return bool(data["hypothesis_answer"])
                    except json.JSONDecodeError:
                        pass
                    break

            potential_json_start = start_idx + 1

        raise ValueError("No valid JSON with hypothesis_answer found")
    except Exception as e:
        raise RuntimeError(f"Failed to extract hypothesis answer: {e}")

def extract_initial_construct_json(answer: str) -> dict[str, Any]:
    """
    提取新版 InitialConstructStage 返回的 JSON 格式:
    {
      "nodes": [
        {"node": "node symbol", "meaning": "...", "role": "..."},
        ...
      ],
      "edges": [
        {"from": "source", "to": "target", "type": "directed/undirected"},
        ...
      ],
      "causal_question": "Is E -> B?"
    }
    """
    try:
        import json
        import re

        # 查找JSON代码块
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'({.*})', answer, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                raise ValueError("No JSON found in the answer.")

        json_str = re.sub(r'//.*', '', json_str)
        data = json.loads(json_str)

        result = {}

        # 提取nodes
        if "nodes" in data and isinstance(data["nodes"], list):
            result["nodes"] = data["nodes"]

        # 提取edges
        if "edges" in data and isinstance(data["edges"], list):
            result["edges"] = data["edges"]

        # 提取causal_question
        if "causal_question" in data:
            result["causal_question"] = data["causal_question"]

        return result

    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing failed: {e}")
        logging.debug(f"Problematic JSON string: {answer[:500]}...")
        raise RuntimeError(f"Failed to parse JSON response: {e}")
    except Exception as e:
        logging.error(f"Failed to extract initial construct structure: {e}")
        logging.debug(f"Problematic response excerpt: {answer[:500]}...")
        raise RuntimeError(f"Failed to extract initial construct structure: {e}")


def convert_edge_format(edges_data: list) -> list:
    """
    将边数据转换为统一格式
    
    输入可能是:
    [{"source": "A", "target": "B", "relation_type": "causal"}, ...]
    或
    [["A", "B"], ["C", "D"], ...]
    
    输出统一为:
    [{"source": "A", "target": "B", "relation_type": "..."}, ...]
    """
    result = []
    
    for edge in edges_data:
        if isinstance(edge, dict):
            # 已经是正确格式
            if "source" in edge and "target" in edge:
                result.append(edge)
        elif isinstance(edge, list) and len(edge) == 2:
            # 转换列表格式为字典格式
            result.append({
                "source": edge[0],
                "target": edge[1],
                "relation_type": "unknown"
            })
        else:
            logging.warning(f"Unexpected edge format: {edge}")
    
    return result
