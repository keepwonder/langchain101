from dotenv import load_dotenv

from langchain import hub
from langchain.agents import (
    AgentExecutor, 
    create_openai_functions_agent,
    create_openai_tools_agent,
    create_xml_agent,
    create_structured_chat_agent,
    create_json_chat_agent,
    create_react_agent,
    create_self_ask_with_search_agent
    )
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatAnthropic


# 加载环境
load_dotenv()

# 初始化tools
tools = [TavilySearchResults(max_results=1)]

# 创建Agent

## 从hub拉取prompt
prompt = hub.pull('hwchase17/openai-functions-agent')

## 定义llm
llm = ChatOpenAI(model='gpt-3.5-turbo-1106')

## 创建agent
agent = create_openai_functions_agent(llm, tools, prompt)

# 运行agent
agent_excutor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_excutor.invoke({'input': 'what is LangChain?'})

print(f'result:{result}')


