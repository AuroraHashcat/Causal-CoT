from SPARQLWrapper import SPARQLWrapper, JSON, POST
import logging
from typing import List, Dict
import time
import json


class CNKnowledgeGraphClient:
    def __init__(self):
        # 先初始化日志，确保在后续操作中可用
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("CNKnowledgeGraphClient")

        # 国内可用的知识图谱端点（按优先级排序）
        self.endpoints = [
            "https://api.ownthink.com/bot/knowledge?platform=sparql",  # 思知知识图谱
            "http://dbpedia.org/sparql",  # DBPedia备用
            "https://krr.triply.cc/krr/sparql"  # 额外备用端点
        ]
        self.current_endpoint = None
        self.sparql = None
        self._init_sparql()  # 初始化并测试端点

    def _init_sparql(self):
        """初始化SPARQL客户端，自动切换可用端点"""
        for endpoint in self.endpoints:
            try:
                # 测试端点连接
                test_sparql = SPARQLWrapper(endpoint)
                # 特殊处理思知知识图谱的查询方式
                if "ownthink" in endpoint:
                    test_sparql.setMethod(POST)

                # 简单的测试查询
                test_query = """
                SELECT ?s ?p ?o WHERE {
                    ?s ?p ?o .
                } LIMIT 1
                """
                test_sparql.setQuery(test_query)
                test_sparql.setReturnFormat(JSON)

                # 执行查询并处理结果（兼容不同版本SPARQLWrapper）
                result = test_sparql.query()

                # 处理不同端点的响应
                if "ownthink" in endpoint:
                    # 思知知识图谱需要特殊处理
                    try:
                        # 尝试新版本接口
                        content = result.response.read().decode('utf-8')
                    except AttributeError:
                        # 尝试旧版本接口
                        content = result.read().decode('utf-8')

                    json_data = json.loads(content)
                    if "data" not in json_data:
                        raise Exception("思知知识图谱返回格式不正确")
                else:
                    # 标准SPARQL端点处理
                    json_data = result.convert()

                # 连接成功，使用该端点
                self.current_endpoint = endpoint
                self.sparql = SPARQLWrapper(endpoint)
                if "ownthink" in endpoint:
                    self.sparql.setMethod(POST)
                self.sparql.setReturnFormat(JSON)
                self.logger.info(f"已连接到知识图谱端点: {endpoint}")
                return
            except Exception as e:
                self.logger.warning(f"端点 {endpoint} 不可用: {e}")
                continue

        # 所有端点都不可用时，抛出警告但不崩溃
        self.logger.error("所有知识图谱端点均不可用，将返回空结果")
        self.sparql = None

    def _retry_with_endpoint_switch(self, func, *args, **kwargs):
        """带端点切换的重试机制"""
        max_retries = 2
        for attempt in range(max_retries + 1):
            if not self.sparql:
                self._init_sparql()  # 尝试重新初始化

            if self.sparql:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.logger.error(f"操作失败（尝试 {attempt + 1}/{max_retries + 1}）: {e}")
                    # 切换到下一个端点
                    current_idx = self.endpoints.index(self.current_endpoint)
                    next_idx = (current_idx + 1) % len(self.endpoints)
                    self.current_endpoint = self.endpoints[next_idx]
                    self.sparql = SPARQLWrapper(self.current_endpoint)
                    if "ownthink" in self.current_endpoint:
                        self.sparql.setMethod(POST)
                    self.sparql.setReturnFormat(JSON)
                    time.sleep(1)  # 等待1秒再重试
            else:
                break

        self.logger.error("达到最大重试次数，返回空结果")
        return None if func.__name__ == "get_entity_id" else []

    def get_entity_id(self, entity_name: str, lang: str = "zh") -> str:
        """通过实体名称获取知识图谱ID（带重试和多语言支持）"""

        def _inner():
            lang_code = "zh" if lang == "zh" else "en"

            # 思知知识图谱查询
            if "ownthink" in self.current_endpoint:
                query = f"""
                SELECT ?s WHERE {{
                    ?s rdfs:label "{entity_name}"@{lang_code}.
                }}
                LIMIT 1
                """
            else:  # DBPedia查询 - 修复UNION语法
                query = f"""
                SELECT ?s WHERE {{
                    {{?s rdfs:label "{entity_name}"@{lang_code}.}}
                    UNION
                    {{?s skos:altLabel "{entity_name}"@{lang_code}.}}
                }}
                LIMIT 1
                """

            self.sparql.setQuery(query)
            result = self.sparql.query()

            print("\n\n\n\n\n\nResult: ", result)

            # 处理不同端点的响应格式
            if "ownthink" in self.current_endpoint:
                # 思知知识图谱返回特殊格式的JSON
                try:
                    # 尝试新版本接口
                    content = result.response.read().decode('utf-8')
                except AttributeError:
                    # 尝试旧版本接口
                    content = result.read().decode('utf-8')

                json_data = json.loads(content)
                if "data" in json_data and len(json_data["data"]) > 0:
                    return json_data["data"][0]["s"]["value"].split("/")[-1]
            else:
                # 标准SPARQL响应处理
                json_data = result.convert()
                if json_data["results"]["bindings"]:
                    return json_data["results"]["bindings"][0]["s"]["value"].split("/")[-1]

            # 中文清洗后重试
            if lang == "zh":
                en_result = self.get_entity_id(entity_name, lang="en")
                if en_result:
                    return en_result
                self.logger.info(f"英文查询未匹配到实体 '{entity_name}'，返回空")
                return None
            return None

        return self._retry_with_endpoint_switch(_inner)

    def find_mediators(self, entity1: str, entity2: str) -> List[Dict]:
        """查找两个实体间的中介实体（带重试）"""

        def _inner():
            e1_id = self.get_entity_id(entity1)
            e2_id = self.get_entity_id(entity2)
            if not e1_id or not e2_id:
                return []

            # 构建实体URI
            if "ownthink" in self.current_endpoint:
                e1_uri = f"http://www.ownthink.com/resource/{e1_id}"
                e2_uri = f"http://www.ownthink.com/resource/{e2_id}"
                query = f"""
                SELECT ?mediator ?mediatorLabel ?rel1 ?rel2
                WHERE {{
                    <{e1_uri}> ?rel1 ?mediator.
                    ?mediator ?rel2 <{e2_uri}>.
                    ?mediator rdfs:label ?mediatorLabel.
                    FILTER (lang(?mediatorLabel) = "zh" || lang(?mediatorLabel) = "en")
                }}
                LIMIT 10
                """
            else:  # DBPedia及其他标准端点
                e1_uri = f"http://dbpedia.org/resource/{e1_id}"
                e2_uri = f"http://dbpedia.org/resource/{e2_id}"
                query = f"""
                SELECT ?mediator ?mediatorLabel ?rel1 ?rel1Label ?rel2 ?rel2Label
                WHERE {{
                    <{e1_uri}> ?rel1 ?mediator.
                    ?mediator ?rel2 <{e2_uri}>.
                    ?mediator rdfs:label ?mediatorLabel.
                    ?rel1 rdfs:label ?rel1Label.
                    ?rel2 rdfs:label ?rel2Label.
                    FILTER (lang(?mediatorLabel) = "zh" || lang(?mediatorLabel) = "en")
                }}
                LIMIT 10
                """

            self.sparql.setQuery(query)
            results = self._get_query_results()
            return self._parse_mediator_results(results)

        return self._retry_with_endpoint_switch(_inner)

    def find_common_causes(self, entity1: str, entity2: str) -> List[Dict]:
        """查找两个实体的共同原因（带重试）"""

        def _inner():
            e1_id = self.get_entity_id(entity1)
            e2_id = self.get_entity_id(entity2)
            if not e1_id or not e2_id:
                return []

            # 构建实体URI
            if "ownthink" in self.current_endpoint:
                e1_uri = f"http://www.ownthink.com/resource/{e1_id}"
                e2_uri = f"http://www.ownthink.com/resource/{e2_id}"
                query = f"""
                SELECT ?cause ?causeLabel ?rel1 ?rel2
                WHERE {{
                    ?cause ?rel1 <{e1_uri}>.
                    ?cause ?rel2 <{e2_uri}>.
                    ?cause rdfs:label ?causeLabel.
                    FILTER (lang(?causeLabel) = "zh" || lang(?causeLabel) = "en")
                }}
                LIMIT 10
                """
            else:  # DBPedia及其他标准端点
                e1_uri = f"http://dbpedia.org/resource/{e1_id}"
                e2_uri = f"http://dbpedia.org/resource/{e2_id}"
                query = f"""
                SELECT ?cause ?causeLabel ?rel1 ?rel1Label ?rel2 ?rel2Label
                WHERE {{
                    ?cause ?rel1 <{e1_uri}>.
                    ?cause ?rel2 <{e2_uri}>.
                    ?cause rdfs:label ?causeLabel.
                    ?rel1 rdfs:label ?rel1Label.
                    ?rel2 rdfs:label ?rel2Label.
                    FILTER (lang(?causeLabel) = "zh" || lang(?causeLabel) = "en")
                }}
                LIMIT 10
                """

            self.sparql.setQuery(query)
            results = self._get_query_results()
            return self._parse_common_cause_results(results)

        return self._retry_with_endpoint_switch(_inner)

    def _get_query_results(self):
        """统一处理不同端点的查询结果格式"""
        result = self.sparql.query()
        if "ownthink" in self.current_endpoint:
            # 处理思知知识图谱的响应
            try:
                # 尝试新版本接口
                content = result.response.read().decode('utf-8')
            except AttributeError:
                # 尝试旧版本接口
                content = result.read().decode('utf-8')

            json_data = json.loads(content)
            # 转换为标准SPARQL结果格式
            return {"results": {"bindings": json_data.get("data", [])}}
        else:
            # 标准SPARQL端点处理
            return result.convert()

    def _parse_mediator_results(self, results: Dict) -> List[Dict]:
        parsed = []
        for res in results["results"]["bindings"]:
            # 适配不同知识图谱的返回格式
            rel1 = res.get("rel1Label", res.get("rel1", {})).get("value", "").split("/")[-1]
            rel2 = res.get("rel2Label", res.get("rel2", {})).get("value", "").split("/")[-1]
            parsed.append({
                "mediator": res.get("mediatorLabel", {}).get("value", ""),
                "relation_from_entity1": rel1,
                "relation_to_entity2": rel2
            })
        return parsed

    def _parse_common_cause_results(self, results: Dict) -> List[Dict]:
        parsed = []
        for res in results["results"]["bindings"]:
            rel1 = res.get("rel1Label", res.get("rel1", {})).get("value", "").split("/")[-1]
            rel2 = res.get("rel2Label", res.get("rel2", {})).get("value", "").split("/")[-1]
            parsed.append({
                "common_cause": res.get("causeLabel", {}).get("value", ""),
                "relation_to_entity1": rel1,
                "relation_to_entity2": rel2
            })
        return parsed