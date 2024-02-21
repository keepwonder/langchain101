# LangChain 101: 06. 链（Chains）

`LangChain`在今年的1月8号发布了v0.1.0版本。之前也断断续续的学习了一遍，奈何学了忘，忘了学。今天开始重新整理一遍，顺便记录下来，顺手写一个【LangChain极简入门课】，供小白使用（大佬可以跳过）。
本教程默认以下前提：
- 使用Python版本的LangChain
- LLM使用OpenAI的gpt-3.5-turbo-1106
- LangChain发展非常迅速，虽然已经大版本v0.1了，后续版本肯定会继续迭代，为避免教程中代码失效。本教程统一使用版本 **0.1.2**

根据Langchain的[代码约定](https://github.com/langchain-ai/langchain/blob/v0.1.2/pyproject.toml#L11)，Python版本 ">=3.8.1,<4.0"。

所有代码和教程开源在github：[https://github.com/keepwonder/langchain101](https://github.com/keepwonder/langchain101)

## 什么是链？

`链`指的是一个调用序列——无论是对 LLM、工具还是数据预处理步骤。主要方法是使用 LCEL。

`LangChain` 目前支持两种类型的链:
- 使用LCEL创建的链。在这种情况下，LangChain 提供了一种更高级的构造函数方法。但是，所有这些都是通过 `LCEL`来构建链的。
- [遗留]链由遗留 Chain 类的子类构造。这些链并不在底层使用 LCEL，而是相当独立的类。

`LangChain`官方正在将所有链改造成通过LCEL来创建。基于以下几个原因：
1. 使用这种方式创建链，可以通过修改LCEL来修改链的内部
2. 这些链原生支持流、异步和即时批处理。
3. 这些链可以自动获取观察每一步的运行过程。

### LCEL 链
- **create_stuff_documents_chain**
- create_openai_fn_runnable
- create_structured_output_runnable	
- load_query_constructor_runnable
- create_sql_query_chain
- create_history_aware_retriever
- create_retrieval_chain

### 遗留的链
- APIChain
- OpenAPIEndpointChain
- ConversationalRetrievalChain
- StuffDocumentsChain
- ReduceDocumentsChain
- MapReduceDocumentsChain
- RefineDocumentsChain
- MapRerankDocumentsChain
- ConstitutionalChain
- **LLMChain**
- ElasticsearchDatabaseChain
- FlareChain
- ArangoGraphQAChain
- GraphCypherQAChain
- FalkorDBGraphQAChain
- HugeGraphQAChain
- KuzuQAChain
- NebulaGraphQAChain
- NeptuneOpenCypherQAChain
- GraphSparqlChain
- LLMMath
- LLMCheckerChain
- LLMSummarizationChecker
- create_citation_fuzzy_match_chain
- create_extraction_chain
- create_extraction_chain_pydantic
- get_openapi_chain
- create_qa_with_structure_chain
- create_qa_with_sources_chain
- QAGenerationChain
- RetrievalQAWithSourcesChain
- load_qa_with_sources_chain
- RetrievalQA
- MultiPromptChain
- MultiRetrievalQAChain
- EmbeddingRouterChain
- LLMRouterChain
- load_summarize_chain
- LLMRequestsChain

## 示例
### 最基础的链 LLMchain
作为入门教程，我们从最基础的概念开始。`LLMChain`是`LangChain`中最基础的链。

`LLMChain`接受如下组件：
- LLM
- 提示词模板

在前面的教程中，我们学习了OpenAI LLM的使用。今天我们基于OpenAI LLM，利用`LLMChain`构建自己的第一个链。

0. 环境准备
   
```python
from dotenv import load_dotenv
load_dotenv()
```

1. 准备组件
   
```python
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0)
prompt = PromptTemplate(
    input_variables=["color"],
    template="What is the hex code of color {color}?",
)
```

2. 基于组件创建`LLMChain`实例
   
我们要创建的链，基于提示词模版，提供基于颜色名字询问对应的16进制代码的能力。

```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
```

3. 基于链提问
   
现在我们利用创建的 `LLMChain` 实例提问。注意，提问中我们只需要提供第一步中创建的提示词模版变量的值。我们分别提问green，cyan，magento三种颜色的16进制代码。

```python
print(chain.invoke({'color':'red'}))
print(chain.invoke({'color':'green'}))
print(chain.invoke({'color':'blue'}))
```

你应该得到如下输出：

```shell
{'color': 'red', 'text': '\n\nThe hex code for red is #FF0000.'}
{'color': 'green', 'text': '\n\nThe hex code for green is #00FF00.'}
{'color': 'blue', 'text': '\n\nThe hex code for blue is #0000FF.'}
```

### 基于LCEL创建链
下面是基于`create_stuff_documents_chain`的示例

这个链需要提供一个文档列表，并将它们全部格式进一个提示词，然后将该提示词传递给 LLM。它传递所有文档，我们需要确保使用的LLM的上下文窗口大小满足。

```python
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_messages(
    [("system", "What are everyone's favorite colors:\n\n{context}")]
)
llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
chain = create_stuff_documents_chain(llm, prompt)

docs = [
    Document(page_content="Jesse loves red but not yellow"),
    Document(page_content = "Jamal loves green but not as much as he loves orange")
]

chain.invoke({"context": docs})
```

类似输出如下
```shell
'Amy loves blue but not purple\n\nSara loves pink but not brown\n\nEmma loves purple but not blue\n\nMichael loves black but not white\n\nSophia loves yellow but not green\n\nLiam loves orange but not red\n\nOlivia loves brown but not pink'
```

## 总结
本节课程中，我们学习了`LangChain` 提出的最重要的概念 - 链（`Chain`） ，介绍了如何使用最基础的LLMChain和基于LCEL创建链。

本节课程的完整示例代码，请参考 [chains.ipynb](./chains.ipynb)。

### 相关文档
1. [https://python.langchain.com/docs/modules/chains](https://python.langchain.com/docs/modules/chains) 
