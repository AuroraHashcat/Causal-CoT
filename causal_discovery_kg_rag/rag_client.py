# -*- coding: utf-8 -*-
"""
OnlineRAGClient: 纯在线检索的 RAG 客户端（无需本地模型、无需本地语料）
数据源：OpenAlex、PubMed、arXiv（免 Key 公共 API；一般可在大陆直接访问）
功能：多源检索 -> 结果清洗/排序 -> 片段抽取 -> 简单因果关系抽取（启发式）
依赖：
    pip install requests
    # 可选重排（推荐）：
    pip install rank_bm25

使用示例：
    rag = OnlineRAGClient(max_search_results=5)
    out = rag.rag_search("HVAC energy efficiency control causes comfort issues")
    print(out["rag_contexts"][:2])
    print(out["causal_relations"][:3])
"""
import logging
import re
import time
import html
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import requests

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False


# ---------------------------
# 公共工具
# ---------------------------

def _now_ts() -> int:
    return int(time.time())


def _safe_get(d: dict, key: str, default: str = "") -> str:
    v = d.get(key, default)
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def _truncate(s: str, n: int = 500) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + "…"


def _split_sentences(text: str) -> List[str]:
    # 粗略句子切分（中英文混合）
    text = re.sub(r"\s+", " ", text)
    # 以常见标点切分
    parts = re.split(r"(?<=[。；;.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _keyword_tokens(s: str) -> List[str]:
    s = s.lower()
    # 简易分词：按非字母数字分割
    toks = re.split(r"[^a-z0-9_]+", s)
    return [t for t in toks if t]


# ---------------------------
# 数据结构
# ---------------------------

@dataclass
class RetrievedDoc:
    title: str
    contents: str
    url: str = ""
    source: str = ""
    published: str = ""
    score: float = 0.0  # 排序用


# ---------------------------
# Provider 基类与实现
# ---------------------------

class BaseProvider:
    name: str = "base"

    def search(self, query: str, num: int = 5) -> List[RetrievedDoc]:
        raise NotImplementedError


class OpenAlexProvider(BaseProvider):
    """OpenAlex: https://api.openalex.org/works?search=..."""
    name = "openalex"
    API = "https://api.openalex.org/works"

    def search(self, query: str, num: int = 5) -> List[RetrievedDoc]:
        params = {
            "search": query,
            "per_page": min(max(num, 1), 25),
            "sort": "relevance_score:desc"
        }
        r = requests.get(self.API, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("results", []):
            title = _safe_get(item, "title")
            # 优先摘要，否则拼字段
            abstract = _safe_get(item, "abstract_inverted_index", "")
            if isinstance(item.get("abstract_inverted_index"), dict):
                # 反向索引抽回文本
                inv = item["abstract_inverted_index"]
                max_pos = max([max(v) for v in inv.values()]) if inv else -1
                restored = [""] * (max_pos + 1 if max_pos >= 0 else 0)
                for word, poss in inv.items():
                    for p in poss:
                        if p < len(restored):
                            restored[p] = word
                abstract = " ".join(restored)
            url = ""
            # 获取最佳链接
            host_venue = item.get("host_venue") or {}
            url = host_venue.get("url") or item.get("id", "")
            published = (item.get("publication_year") and str(item["publication_year"])) or ""
            authors = []
            for au in item.get("authorships", []):
                nm = au.get("author", {}).get("display_name")
                if nm:
                    authors.append(nm)
            meta = f"Authors: {', '.join(authors)}. Year: {published}."
            contents = (title + ". " + abstract + " " + meta).strip()
            results.append(RetrievedDoc(
                title=title or "No Title",
                contents=contents,
                url=url,
                source=self.name,
                published=str(published)
            ))
        return results


class PubMedProvider(BaseProvider):
    """PubMed E-utilities:
       esearch: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
       esummary: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi
    """
    name = "pubmed"
    ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def search(self, query: str, num: int = 5) -> List[RetrievedDoc]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": min(max(num, 1), 20),
            "sort": "relevance"
        }
        r = requests.get(self.ESEARCH, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        params2 = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json"
        }
        r2 = requests.get(self.ESUMMARY, params=params2, timeout=15)
        r2.raise_for_status()
        data2 = r2.json()
        result = []
        ulist = data2.get("result", {})
        for pid in ids:
            rec = ulist.get(pid, {})
            title = _safe_get(rec, "title")
            snip = _safe_get(rec, "sortfirstauthor", "")
            journal = _safe_get(rec, "fulljournalname", "")
            pubdate = _safe_get(rec, "pubdate", "")
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
            contents = f"{title}. {journal}. {pubdate}. First author: {snip}."
            result.append(RetrievedDoc(
                title=title or "No Title",
                contents=contents,
                url=url,
                source=self.name,
                published=pubdate
            ))
        return result


class ArxivProvider(BaseProvider):
    """arXiv API: http://export.arxiv.org/api/query?search_query=..."""
    name = "arxiv"
    API = "http://export.arxiv.org/api/query"

    def search(self, query: str, num: int = 5) -> List[RetrievedDoc]:
        # arXiv 是 Atom XML；我们做简单解析
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(max(num, 1), 20),
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        r = requests.get(self.API, params=params, timeout=20)
        r.raise_for_status()
        text = r.text
        # 解析每个 entry 的 title、summary、link、published
        entries = re.split(r"<entry>", text)[1:]
        results = []
        for e in entries:
            title_m = re.search(r"<title>(.*?)</title>", e, re.S)
            summary_m = re.search(r"<summary>(.*?)</summary>", e, re.S)
            link_m = re.search(r'<link .*?href="(http[^"]+)"', e)
            pub_m = re.search(r"<published>(.*?)</published>", e)

            title = html.unescape((title_m.group(1) if title_m else "").strip())
            summary = html.unescape((summary_m.group(1) if summary_m else "").strip())
            url = link_m.group(1) if link_m else ""
            published = pub_m.group(1) if pub_m else ""
            contents = f"{title}. {summary}"
            results.append(RetrievedDoc(
                title=title or "No Title",
                contents=contents,
                url=url,
                source=self.name,
                published=published
            ))
        return results


# ---------------------------
# OnlineRAGClient 主体
# ---------------------------

class OnlineRAGClient:
    """
    仅使用在线数据源进行 RAG 检索与因果抽取：
      - 不加载本地语料
      - 不加载/调用本地向量模型
      - 可扩展 Provider
    输出：
      - rag_contexts: 标准化的检索结果
      - causal_relations: 简易因果抽取的三元组
    """

    def __init__(self,
                 max_search_results: int = 5,
                 providers: Optional[List[BaseProvider]] = None,
                 enable_bm25_rerank: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        参数:
            max_search_results: 最终合并后的最大结果数
            providers: 可注入 Provider 列表；默认使用 OpenAlex/PubMed/arXiv
            enable_bm25_rerank: 是否使用 BM25 重排（安装 rank_bm25 时生效）
        """
        self.max_results = max_search_results
        self.logger = logger or logging.getLogger("OnlineRAGClient")
        self.logger.setLevel(logging.INFO)
        self.providers = providers or [
            OpenAlexProvider(),
            PubMedProvider(),
            ArxivProvider(),
        ]
        self.enable_bm25_rerank = enable_bm25_rerank and _HAS_BM25

        # 因果触发词与抽取规则配置
        self.causal_cues = [
            " because ", " due to ", " leads to ", " lead to ", " result in ", " results in ",
            " cause ", " causes ", " caused ", " therefore ", " thus ", " as a result "
        ]

    # ======= 对外入口 =======
    def rag_search(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        在线检索 + 简易因果抽取
        返回：
        {
            "rag_contexts": [
                {"title":..., "snippet":..., "url":..., "source":..., "published":...},
                ...
            ],
            "causal_relations": [
                {"subject":..., "relation":"causes", "object":..., "evidence":..., "source":..., "url":...},
                ...
            ]
        }
        """
        try:
            docs = self._multi_source_search(query, self.max_results)
            rag_contexts = [{
                "title": d.title or "No Title",
                "snippet": _truncate(d.contents, 600),
                "url": d.url,
                "source": d.source,
                "published": d.published
            } for d in docs]

            relations = self._extract_causal_relations(query, docs)
            return {
                "rag_contexts": rag_contexts,
                "causal_relations": relations
            }
        except Exception as e:
            self.logger.error(f"Online RAG search failed: {e}", exc_info=True)
            return {"rag_contexts": [], "causal_relations": []}

    # ======= 检索与排序 =======
    def _multi_source_search(self, query: str, k: int) -> List[RetrievedDoc]:
        all_docs: List[RetrievedDoc] = []
        for p in self.providers:
            try:
                res = p.search(query, num=max(k, 3))
                self.logger.info(f"[{p.name}] returned {len(res)} items")
                all_docs.extend(res)
            except Exception as e:
                self.logger.warning(f"Provider {p.name} error: {e}")

        if not all_docs:
            return []

        # 去重（按标题+URL）
        uniq: Dict[str, RetrievedDoc] = {}
        for d in all_docs:
            key = (d.title.strip().lower() + "||" + d.url.strip().lower())
            if key not in uniq:
                uniq[key] = d

        docs = list(uniq.values())

        # 排序：BM25（若可用）/ 否则用关键词交集数量
        docs = self._rerank(query, docs)
        return docs[:k]

    def _rerank(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        if self.enable_bm25_rerank and docs:
            corpus = [d.title + " " + d.contents for d in docs]
            tokenized_corpus = [_keyword_tokens(c) for c in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            q_tokens = _keyword_tokens(query)
            scores = bm25.get_scores(q_tokens)
            for d, s in zip(docs, scores):
                d.score = float(s)
            docs.sort(key=lambda x: x.score, reverse=True)
            return docs

        # 退化排序：按query关键词交集个数
        qset = set(_keyword_tokens(query))
        for d in docs:
            dset = set(_keyword_tokens(d.title + " " + d.contents))
            d.score = float(len(qset & dset))
        docs.sort(key=lambda x: x.score, reverse=True)
        return docs

    # ======= 因果抽取 =======
    def _extract_causal_relations(self, query: str, docs: List[RetrievedDoc]) -> List[Dict[str, Any]]:
        """
        启发式因果抽取（不调用 LLM）：
        - 在句子中查找因果触发词
        - 触发词前的片段 -> subject（精简）
        - 触发词后的片段 -> object（精简）
        - relation 固定为 "causes"
        """
        results: List[Dict[str, Any]] = []
        for d in docs:
            sents = _split_sentences(d.contents)
            for sent in sents:
                sent_l = " " + sent.lower() + " "
                cue_pos = None
                cue = None
                for c in self.causal_cues:
                    idx = sent_l.find(c)
                    if idx != -1:
                        cue_pos = idx
                        cue = c
                        break
                if cue_pos is None:
                    continue

                # 主体与客体抽取
                subj = sent[:max(0, cue_pos)].strip()
                obj = sent[max(0, cue_pos + len(cue)):].strip()

                # 清洗主体/客体：去掉过长/过短
                subj = self._clean_phrase(subj)
                obj = self._clean_phrase(obj)

                if subj and obj:
                    results.append({
                        "subject": subj,
                        "relation": "causes",
                        "object": obj,
                        "evidence": _truncate(sent, 300),
                        "source": d.source,
                        "url": d.url
                    })
        return results

    @staticmethod
    def _clean_phrase(text: str, max_len: int = 120) -> str:
        # 去冗余标点与引号
        text = text.strip().strip(",;:，；：.。\"'“”‘’()[]{}")
        # 限长
        return _truncate(text, max_len)


# ---------------------------
# 向后兼容封装（可选）
# ---------------------------

class RAGClient(OnlineRAGClient):
    """
    向后兼容您原始类名 RAGClient，但内部改为 OnlineRAGClient 的在线实现。
    - 保留 rag_search(query) 的签名与返回结构。
    - 去除本地 FlashRAG、PromptTemplate、Config 等依赖。
    """
    def __init__(self,
                 max_search_results: int = 5):
        super().__init__(max_search_results=max_search_results)