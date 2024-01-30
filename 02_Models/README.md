# LangChain 101: 02. 模型

`LangChain`在今年的1月8号发布了v0.1.0版本。之前也断断续续的学习了一遍，奈何学了忘，忘了学。今天开始重新整理一遍，顺便记录下来，顺手写一个【LangChain极简入门课】，供小白使用（大佬可以跳过）。
本教程默认以下前提：
- 使用Python版本的LangChain
- LLM使用OpenAI的gpt-3.5-turbo-1106
- LangChain发展非常迅速，虽然已经大版本v0.1了，后续版本肯定会继续迭代，为避免教程中代码失效。本教程统一使用版本 **0.1.2**

根据Langchain的[代码约定](https://github.com/langchain-ai/langchain/blob/v0.0.235/pyproject.toml#L14C1-L14C24)，Python版本 ">=3.8.1,<4.0"。

所有代码和教程开源在github：[https://github.com/keepwonder/langchain101](https://github.com/keepwonder/langchain101)

----
![](./model_io.jpg)

## Model I/O
今天我们开始介绍Model I/O模块，如上图所示，Model I/O模块包含3部分， 左边绿色部分  Format，中间紫色部分 Predict，右边蓝色部分 Parse。而这三部分分别对应LangChain中最基础也是最重要的三个概念

|输入|处理|输出|
|----|----|---|
|Prompts|LLMs / Chat Models|Output Parsers|

今天我们要介绍的是处理部分，也就是模型。

## 模型简介
任何语言模型应用程序的核心元素都是：模型。LangChain 为我们提供了与任何语言模型接口的构件。
LangChain为我们封装了两类模型：
- 大语言模型（LLM）
- 聊天模型（Chat Models）

### 大语言模型（LLM）
大语言模型(LLM)是 LangChain 的核心组成部分。LangChain为许多不同的 LLM 进行交互提供一个标准接口。具体来说，这个接口接受一个字符串作为输入并返回一个字符串。

有很多 LLM 提供者(OpenAI、 Cohere、 Hugging Face 等)—— `LLM` 类被设计为为所有这些提供者提供一个标准接口。

### 聊天模型（Chat Models）
聊天模型(Chat Models)是 LangChain 的核心组件。LangChain 不提供自己的聊天模型，而是提供一个与许多不同模型交互的标准接口。具体来说，这个接口接受一个消息列表作为输入并返回一条消息。

有很多模型提供者(OpenAI、 Cohere、 Hugging Face 等)—— `ChatModel` 类被设计用来为所有这些提供者提供一个标准接口。

## LangChain 与 OpenAI 模型
查看[OpenAI-models官方文档](https://platform.openai.com/docs/models)，我们可以看到，OpenAI提供了很多种类的模型，其中我们使用到的就是GPT系列。

Langchain提供接口集成不同的模型。为了便于切换模型，Langchain将不同模型抽象为相同的接口 `BaseLanguageModel`，并实现`Runnable`接口，通过LCEL提供 `invoke`、`stream`、`batch`、`ainvoke`、`astream`、`abatch` 、`astream_log`函数来调用模型。

不管使用LLM还是聊天模型，我们都可以使用统一的接口来进行预测输出

## 示例代码

### 与 LLM 的交互
[00_LLM.ipynb](./00_LLM.ipynb)

与LLM的交互，我们需要使用 `langchain_openai` 模块中的 `OpenAI` 类。

```python
from langchain_openai import OpenAI

llm = OpenAI()

llm.invoke("What are some theories about the relationship between unemployment and inflation?")
```
你应该能看到类似如下输出：
```shell
"\n\n1. Phillips Curve: This theory, proposed by economist William Phillips, suggests a negative relationship between unemployment and inflation. As the unemployment rate decreases, the demand for labor increases, leading to higher wages and production costs, which in turn leads to higher prices and inflation.\n\n2. Natural Rate of Unemployment: This theory suggests that there is a natural or equilibrium rate of unemployment that is consistent with stable inflation. Any deviation from this rate, either above or below, will result in inflation or deflation respectively.\n\n3. Expectations Theory: According to this theory, inflation is influenced by people's expectations about future price levels. If individuals expect prices to rise, they will demand higher wages, leading to a rise in inflation. Similarly, if they expect prices to fall, they will accept lower wages, leading to a decrease in inflation.\n\n4. Cost-Push Theory: This theory suggests that inflation is caused by an increase in production costs, such as wages, raw material prices, and taxes. As these costs rise, businesses pass on the extra costs to consumers in the form of higher prices, leading to inflation.\n\n5. Demand-Pull Theory: This theory proposes that inflation is caused by an increase in consumer demand, which exceeds the available supply of goods and services. This leads"
```

### 与聊天模型的交互
[01_Chat_Models.ipynb](./01_Chat_Models.ipynb)

与LLM的交互，我们需要使用 `langchain_openai` 模块中的 `ChatOpenAI` 类。

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

chat = ChatOpenAI()

messages = [SystemMessage(content="You're a helpful assistant"),
            HumanMessage(content="What is the purpose of model regularization")]

chat.invoke(messages)

```
你应该能看到类似如下输出：
```shell
AIMessage(content="The purpose of model regularization is to prevent overfitting in machine learning models. Overfitting occurs when a model becomes too complex and starts to memorize the training data instead of learning general patterns. This can lead to poor performance on new, unseen data.\n\nRegularization techniques aim to add a penalty term to the model's objective function, discouraging overly complex or large parameter values. This helps to control the model's complexity and reduce the likelihood of overfitting. Regularization can also help with feature selection by shrinking less important features towards zero, effectively removing them from the model.\n\nCommon regularization techniques include L1 regularization (Lasso), L2 regularization (Ridge), and dropout regularization. These techniques help to improve a model's generalization ability, making it more robust and reliable when applied to new data.")

```

### 消息类
Langchain框架提供了5个消息类，分别是 `AIMessage`、`HumanMessage` 、`SystemMessage`、`FunctionMessage`、`ToolMessage`和`ChatMessage`。它们对应了OpenAI聊天模型API支持的不同角色 `assistant`、`user`、`system`、`tool`和`function`。请参考 [OpenAI API文档 - Chat](https://platform.openai.com/docs/api-reference/chat/create)。

一般我们用户的最多的就`AIMessage`、`HumanMessage`和`SystemMessage`。

| Langchain类 | OpenAI角色 | 作用 |
| -------- | ------- | ------- |
| AIMessage | assistant | 模型回答的消息 |
| HumanMessage | user | 用户向模型的请求或提问 |
| SystemMessage | system | 系统指令，用于指定模型的行为 |


## 总结
本节课程中，我们学习了模型的基本概念，LLM与聊天模型的差异，并基于 `Langchain` 实现了分别与OpenAI LLM和Chat Models的交互。

要注意，虽然是聊天，但是当前我们所实现的交互并没有记忆能力，也就是说，模型并不会记住之前的对话内容。在后续的内容中，我们将学习如何实现记忆能力。

### 相关文档
1. [https://platform.openai.com/docs/models](https://platform.openai.com/docs/**models**)
2. [https://python.langchain.com/docs/modules/model_io/llms/quick_start](https://python.langchain.com/docs/modules/model_io/llms/quick_start)
3. [https://python.langchain.com/docs/modules/model_io/chat/quick_start](https://python.langchain.com/docs/modules/model_io/chat/quick_start)
