from typing import TypedDict,List
from langchain_core.tools import tool

import os,requests
from dotenv import load_dotenv
load_dotenv()


LANGSEARCH_API_KEY=os.getenv('LANGSEARCH_API_KEY')

class SearchResult(TypedDict):
    title:str
    url:str
    snippet:str

@tool
def web_search(query:str,max_results:int=5) -> str:
    """Search the web for current news and information"""
    response = requests.post("https://api.langsearch.com/v1/web-search",headers={
        'Authorization':f"Bearer {LANGSEARCH_API_KEY}",
        "content-type":"application/json",
    },
    json={
        "query":query,
        "max_results":max_results,
        "summary":True,
    },
    timeout=30
    )
    response.raise_for_status()

    data=response.json()
    results=data["data"]["webPages"]["value"]
    formatted=list()

    for r in results:
        formatted.append(
            f"""
                Title: {r['name']}
                URL: {r['url']}
                Summary: {r.get('summary','')}

            """
        )
        return "\n\n".join(formatted)


print(web_search.func(('What is Langgraph')))