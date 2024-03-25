#!/usr/bin/env python
from typing import List
import os
from fastapi import FastAPI
from groq import Groq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langserve import add_routes
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage

os.environ["GROQ_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "http://ai-ol.sns.sohu.com/v1"
os.environ["OPENAI_API_KEY"] = "0d6e513ab4cb43dd911560b08aa69aec"

# 1. Load Retriver
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = JinaEmbeddings(
    jina_api_key="jina_2f109bd96dcd4988a9fcc2962302bb2dqi8vqyi2pWxb7qUSI4EDSU0hAFnE",
    model_name="jina-embeddings-v2-base-en"
)
vector = FAISS.from_documents(documents, embeddings)
retriver = vector.as_retriever()

# 2. Create Tools
retriver_tool = create_retriever_tool(
    retriver,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
# search = TavilySearchResults()
tools = [retriver_tool]#, search]

# 3. Create Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI()  # Groq()
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route
# we need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.
class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"weight": {"type": "chat", "input": "location"}},
    )


class Output(BaseModel):
    output: str


add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
