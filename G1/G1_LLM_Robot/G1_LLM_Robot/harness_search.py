"""统一搜索凭证：显式参数→环境变量→项目tavily_api_key.txt。"""
from pathlib import Path
import os

PROJECT_ROOT=Path(__file__).resolve().parent


def resolve_tavily_key(explicit=None):
    for value in (explicit,os.environ.get('TAVILY_API_KEY')):
        if value and value.strip(): return value.strip()
    path=PROJECT_ROOT/'tavily_api_key.txt'
    if path.is_file():
        value=path.read_text(encoding='utf-8-sig').strip()
        if value: return value
    return None


def search_web(query,max_results=5,api_key=None,client=None):
    if not isinstance(query,str) or not query.strip(): raise ValueError('搜索词不能为空')
    if not 1<=max_results<=10: raise ValueError('max_results:1~10')
    if client is None:
        key=resolve_tavily_key(api_key)
        if not key: raise RuntimeError('未配置Tavily：请在项目根目录tavily_api_key.txt填写密钥，或设置TAVILY_API_KEY')
        from tavily import TavilyClient
        client=TavilyClient(api_key=key)
    return client.search(query=query,search_depth='advanced',max_results=max_results,include_answer=True)


def format_search(response):
    parts=[]
    if response.get('answer'): parts.append('【综合答案】'+response['answer'])
    for row in response.get('results',[]):
        parts.append('- %s\n  %s\n  %s' % (row.get('title',''),(row.get('content') or '')[:400],row.get('url','')))
    return '\n'.join(parts) or '未找到相关结果。'
