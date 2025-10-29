import time
from typing import List, Dict
import logging
import re

class DuckDuckGoSearchClient:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.request_delay = 3.0
        self.last_request_time = 0

    def search(self, query: str) -> List[Dict[str, str]]:
        """使用DuckDuckGo搜索并过滤相关结果"""
        results: List[Dict[str, str]] = []
        
        try:
            from ddgs import DDGS
        except ImportError:
            logging.error("DuckDuckGo search requires ddgs library: pip install ddgs")
            return results
        
        # 实施频率限制
        self._rate_limit()
        
        try:
            logging.info(f"Searching DuckDuckGo for: {query}")
            
            with DDGS(timeout=10) as ddgs:
                search_results = ddgs.text(
                    query,
                    region='wt-wt',
                    safesearch='moderate',
                    timelimit=None,
                    max_results=self.max_results * 2  # 获取更多结果用于过滤
                )
                
                # 处理和过滤搜索结果
                query_keywords = set(query.lower().split())
                
                for result in search_results:
                    if isinstance(result, dict):
                        # 相关性检查
                        if self._is_relevant(result, query_keywords):
                            results.append({
                                'title': result.get('title', ''),
                                'url': result.get('href', ''),
                                'snippet': result.get('body', '')
                            })
                            
                            if len(results) >= self.max_results:
                                break
                
                logging.info(f"DuckDuckGo search completed: {len(results)} relevant results")
                
        except Exception as e:
            logging.debug(f"Search completed with some engine errors (normal): {e}")
        
        return results

    def _is_relevant(self, result: dict, query_keywords: set) -> bool:
        """检查搜索结果是否与查询相关"""
        title = result.get('title', '').lower()
        snippet = result.get('body', '').lower()
        url = result.get('href', '').lower()
        
        # 合并所有文本
        full_text = f"{title} {snippet} {url}"
        result_words = set(re.findall(r'\b\w+\b', full_text))
        
        # 计算关键词重叠度
        overlap = len(query_keywords.intersection(result_words))
        overlap_ratio = overlap / len(query_keywords) if query_keywords else 0
        
        # 过滤明显不相关的结果
        irrelevant_indicators = [
            'erectile dysfunction', 'css', 'javascript', 'html',
            'programming', 'web development', '百度知道', 'zhidao.baidu',
            'general manager', 'managing director'
        ]
        
        for indicator in irrelevant_indicators:
            if indicator in full_text:
                logging.debug(f"Filtered irrelevant result: {indicator} found in {title[:50]}")
                return False
        
        # 要求至少30%的关键词重叠
        return overlap_ratio >= 0.3

    def _rate_limit(self):
        """实施请求频率限制"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            logging.debug(f"Rate limiting: waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
