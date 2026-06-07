from src.llm import get_llm


SYSTEM_PROMPT=f"""
You are a local personal assistant.
Be concise, clear, and helpful.
If context is provided, answer only from that context.
If context is insufficient, say so clearly and answer carefully.
"""

class AssistantAgent:
    def __init__(self,retriever):
        self.retriever=retriever
        self.llm=get_llm()

    def ask_question(self,query:str,top_k:int=4):
        results=self.retriever._retrieve(query=query,top_k=top_k)

        if results:
            context_blocks,sources=[],[]

            for i,item in enumerate(results,start=1):
                content=results.get("content","")
                source=results.get("source","unknown")
                context_blocks.append(f"[{i}] Source:{source}\{content}")
                sources.append(source)

            context="\n\n".join(context_blocks)
            prompt = f"""
            {SYSTEM_PROMPT}

            Context:
            {context}

            User question:
            {query}

            Answer using the context above. If the answer is not supported by the context, say that clearly.
            Also mention the relevant source numbers when useful.
            """

            response=self.llm.invoke(prompt)
            return {
                "answer":response.content,
                "sources":list(dict.fromkeys(sources)),
                "used_rag":True,
            }
        response=self.llm.invoke(f"{SYSTEM_PROMPT}\n\nUser question:{query}")
        return{
            "answer":response.content,
            "sources":[],
            "used_rag":False
        }