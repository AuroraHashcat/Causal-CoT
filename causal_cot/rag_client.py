import requests
import logging
from typing import List, Dict
from ddgs import DDGS  # 开源DuckDuckGo搜索库


class RAGClient:
    def __init__(self, max_search_results: int = 3, kg_endpoint: str = "https://query.wikidata.org/sparql"):
        self.max_results = max_search_results
        self.kg_endpoint = kg_endpoint
        self.ddgs = DDGS()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("RAGClient")

    def hybrid_search(self, query: str) -> Dict[str, List]:
        """混合搜索：结合网页搜索和知识图谱查询"""
        # 1. 网页搜索获取上下文信息
        web_results = self._web_search(query)

        # 2. 知识图谱查询获取结构化关系
        kg_results = self._kg_query(query)

        return {
            "web_contexts": [item["snippet"] for item in web_results],
            "kg_relations": kg_results
        }

    def _web_search(self, query: str) -> List[Dict]:
        """使用DuckDuckGo进行网页搜索"""
        try:
            results = []
            for result in self.ddgs.text(query, max_results=self.max_results * 2):
                if self._is_relevant(result, query):
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("body", ""),
                        "url": result.get("href", "")
                    })
                if len(results) >= self.max_results:
                    break
            self.logger.info(f"Web search for '{query}' returned {len(results)} results")
            return results
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return []

    def _kg_query(self, query: str) -> List[Dict]:
        """查询Wikidata获取实体关系"""
        from SPARQLWrapper import SPARQLWrapper, JSON
        sparql = SPARQLWrapper(self.kg_endpoint)
        sparql.setReturnFormat(JSON)

        # 提取查询中的实体
        entities = self._extract_entities(query)
        if not entities:
            return []

        # 构建SPARQL查询
        sparql_query = f"""
        SELECT ?subjectLabel ?relationLabel ?objectLabel
        WHERE {{
            VALUES ?entity {{ {' '.join(f'wd:{e}' for e in entities)} }}
            ?subject ?relation ?object.
            FILTER (?subject = ?entity || ?object = ?entity)
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "zh,en". }}
        }}
        LIMIT 10
        """

        try:
            sparql.setQuery(sparql_query)
            results = sparql.query().convert()
            return [
                {
                    "subject": res["subjectLabel"]["value"],
                    "relation": res["relationLabel"]["value"],
                    "object": res["objectLabel"]["value"]
                }
                for res in results["results"]["bindings"]
            ]
        except Exception as e:
            self.logger.error(f"KG query failed: {e}")
            return []

    def _extract_entities(self, query: str) -> List[str]:
        """从查询中提取实体并获取Wikidata ID"""
        from SPARQLWrapper import SPARQLWrapper, JSON
        sparql = SPARQLWrapper(self.kg_endpoint)
        sparql.setReturnFormat(JSON)

        entities = []
        for token in query.split():
            if len(token) < 2:
                continue
            sparql.setQuery(f"""
            SELECT ?item WHERE {{
                ?item rdfs:label "{token}"@zh.
                UNION
                ?item rdfs:label "{token}"@en.
            }}
            LIMIT 1
            """)
            try:
                results = sparql.query().convert()
                if results["results"]["bindings"]:
                    entities.append(results["results"]["bindings"][0]["item"]["value"].split("/")[-1])
            except Exception:
                continue
        return entities

    def _is_relevant(self, result: Dict, query: str) -> bool:
        """过滤相关搜索结果"""
        keywords = set(query.lower().split())
        content = f"{result.get('title', '')} {result.get('body', '')}".lower()
        return len(keywords.intersection(content.split())) > 0