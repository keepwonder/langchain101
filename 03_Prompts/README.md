# LangChain 101: 03. 提示词

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

今天我们要介绍的是输入部分，也就是提示词（Prompts）。

## 什么是提示词？
官方：语言模型的提示词是指用户给出的一系列指令或输入，用来指导模型的响应。这有助于模型理解上下文，并生成相关连贯的基于语言的输出，比如回答问题、补全句子或进行对话。
一句话，就像是告诉模型：“请根据这些信息，帮我生成一段文字”。

## 提示词模板

## 总结

### 相关文档