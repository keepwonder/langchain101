# LangChain 101: 03. 提示词

`LangChain`在今年的1月8号发布了v0.1.0版本。之前也断断续续的学习了一遍，奈何学了忘，忘了学。今天开始重新整理一遍，顺便记录下来，顺手写一个【LangChain极简入门课】，供小白使用（大佬可以跳过）。
本教程默认以下前提：
- 使用Python版本的LangChain
- LLM使用OpenAI的gpt-3.5-turbo-1106
- LangChain发展非常迅速，虽然已经大版本v0.1了，后续版本肯定会继续迭代，为避免教程中代码失效。本教程统一使用版本 **0.1.2**

根据Langchain的[代码约定](https://github.com/langchain-ai/langchain/blob/v0.1.2/pyproject.toml#L11)，Python版本 ">=3.8.1,<4.0"。

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

提示词（`prompt`）是指用户向模型提供的一系列指令或输入，用来指导模型做出响应。这有助于模型理解上下文，并生成相关连贯的基于语言的输出，比如回答问题、补全句子或进行对话。
一句话，就像是告诉模型：“请根据这些信息，帮我生成一段文字”。

`LangChain` 提供了一系列的类和函数，简化构建和处理提示词的过程。
- 提示词模板（Prompt Template）：对提示词参数化，提高代码的重用性。
- 聊天提示词模板（Chat Prompt Template）：对聊天消息提示词参数化，提高代码的重用性。
- 样本选择器（Example Selector）：动态选择要包含在提示词中的样本。

## 提示词模板（PromptTemplate）

提示词模板提供了可重用提示词的机制。用户通过传递一组参数给模板，实例化一个提示词。一个提示模板可以包含：
1. 对语言模型的指令
2. 一组少样本示例，以帮助语言模型生成更好的回复
3. 向语言模型提出的问题

一个简单的例子：
```python
from langchain.prompts import PromptTemplate

# template = 'Tell me a {adjective} joke about {content}.'
# prompt_template = PromptTemplate.from_template(template)

# 以上两步可以合并
prompt_template = PromptTemplate.from_template(
    'Tell me a {adjective} joke about {content}.'
)

prompt = prompt_template.format(adjective='funny',content='chickens')

print(f'prompt_template: {prompt_template}')
print(f'prompt: {prompt}')
```

可以看到如下输出

```shell
prompt_template: input_variables=['adjective', 'content'] template='Tell me a {adjective} joke about {content}.'
prompt: Tell me a funny joke about chickens.
```

### 创建模板

`PromptTemplate` 类是 `LangChain` 提供的模版基础类，它接收两个参数：
1. `input_variables` - 输入变量
2. `template` - 模版

模版中通过 `{}` 符号来引用输入变量，比如 `PromptTemplate(input_variables=["content"], template="Tell me a joke about {content}.")`。

模版的实例化通过模板类实例的 `format`函数实现。例子如下：

```python
prompt = PromptTemplate(
    input_variables=["adjecvtive", "content"], 
    template="Tell me a {adjective} joke about {content}."
)
prompt = prompt.format(adjective="funny", content="chickens")
print(prompt)
```
可以看到如下输出
```shell
Tell me a funny joke about chickes.
```

## 聊天提示词模板（ChatPromptTemplate）

聊天模型，比如 `OpenAI` 的GPT模型，接受一系列聊天消息作为输入，每条消息都与一个角色相关联。这个消息列表通常以一定格式串联，构成模型的输入，也就是提示词。

例如，在OpenAI [Chat Completion API](https://platform.openai.com/docs/api-reference/chat/create)中，聊天消息可以与assistant、human或system角色相关联。

为此，LangChain提供了一系列模板，以便更轻松地构建和处理提示词。建议在与聊天模型交互时优先选择使用这些与聊天相关的模板，而不是基础的PromptTemplate，以充分利用框架的优势，提高开发效率。`SystemMessagePromptTemplate`, `AIMessagePromptTemplate`, `HumanMessagePromptTemplate` 是分别用于创建不同角色提示词的模板。

```python
from langchain.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

system_template = "You are a helpful AI bot. Your name is {name}."
system_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template1 = "Hello, how are you doing?"
human_prompt1 = HumanMessagePromptTemplate.from_template(human_template1)

ai_template = "I'm doing well, thanks!"
ai_prompt = AIMessagePromptTemplate.from_template(ai_template)

human_template2 = "{user_input}"
human_prompt2 = HumanMessagePromptTemplate.from_template(human_template2)

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt1, ai_prompt, human_prompt2])
messages = chat_prompt.format_messages(
    name='Bob',
    user_input="What's your name?"
)

print(messages)
```
可以看到如下结果：
```shell
[SystemMessage(content='You are a helpful AI bot. Your name is Bob.'), HumanMessage(content='Hello, how are you doing?'), AIMessage(content="I'm doing well, thanks!"), HumanMessage(content="What's your name?")]
```

也可以简化以上代码：
```python
from langchain.prompts import  ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(name='Bob', user_input='What is your name?')
print(messages)
```
也可以获得同样的结果：
```shell
[SystemMessage(content='You are a helpful AI bot. Your name is Bob.'), HumanMessage(content='Hello, how are you doing?'), AIMessage(content="I'm doing well, thanks!"), HumanMessage(content='What is your name?')]
```

## 样本选择器（Example selectors）

在LLM应用开发中，可能需要从大量样本数据中，选择部分数据包含在提示词中。样本选择器（Example Selector）正是满足该需求的组件，它也通常与少样本提示词配合使用。

`LangChain` 提供了样本选择器的基础接口类 `BaseExampleSelector`，它的实现如下：

```python
class BaseExampleSelector(ABC):
    """Interface for selecting examples to include in prompts."""

    @abstractmethod
    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        
    @abstractmethod
    def add_example(self, example: Dict[str, str]) -> Any:
        """Add new example to store."""
```

每个选择器类必须实现的函数为 `select_examples`，它接受输入变量，然后返回一个示例列表。如何选择这些示例取决于每个特定的实现。

`LangChain` 实现了若干基于不用应用场景或算法的选择器：

|Name|Description|Selector|
|---|---|--|
|Similarity|Uses semantic similarity between inputs and examples to decide which examples to choose.|SemanticSimilarityExampleSelector |
|MMR|Uses Max Marginal Relevance between inputs and examples to decide which examples to choose.|MaxMarginalRelevanceExampleSelector|
|Length|Selects examples based on how many can fit within a certain length|LengthBasedExampleSelector |
|Ngram|Uses ngram overlap between inputs and examples to decide which examples to choose.|NGramOverlapExampleSelector|

### 创建样本

```python
examples = [
    {"input": "hi", "output": "ciao"},
    {"input": "bye", "output": "arrivaderci"},
    {"input": "soccer", "output": "calcio"},
]
```

### 创建一个自定义样本选择器

```python
from langchain_core.example_selectors.base import BaseExampleSelector

class CustomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        new_world = input_variables['input']
        new_world_length = len(new_world)

        best_match = None
        smallest_diff = float('inf')

        for example in self.examples:
            current_diff = abs(len(example['input']) - new_world_length)

            if current_diff < smallest_diff:
                smallest_diff = current_diff
                best_match = example

        return [best_match]
```

实例化选择器
```python
example_selector = CustomExampleSelector(examples)
```

选择样本
```python
example_selector.select_examples({"input": "okay"})
```
输出
```shell
[{'input': 'bye', 'output': 'arrivaderci'}]
```
添加样本
```python
example_selector.add_example({"input": "hand", "output": "mano"})
```

在Prompt中使用
```python
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate

example_prompt = PromptTemplate.from_template("Input: {input} -> Output: {output}")

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Input: {input} -> Output:",
    prefix="Translate the following words from English to Italain:",
    input_variables=['input']
)

print(prompt.format(input='word'))
```

输出
```shell
Translate the following words from English to Italain:

Input: bye -> Output: arrivaderci

Input: word -> Output:
```

### 使用基于长度的样本选择器

```python
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector


example_prompt = PromptTemplate.from_template("Input: {input} \nOutput: {output}")

length_example_selector = LengthBasedExampleSelector(
    # 可选的样本数据
    examples=examples, 
    # 提示词模版
    example_prompt=example_prompt, 
    # 格式化的样本数据的最大长度，通过get_text_length函数来衡量
    max_length=25
)

prompt = FewShotPromptTemplate(
    example_selector=length_example_selector,
    example_prompt=example_prompt,
    suffix="Input: {input} \nOutput:",
    prefix="Translate the following words from English to Italain:",
    input_variables=['input']
)

# 输入量极小，因此所有样本数据都会被选中
print(prompt.format(input='word'))

```
输出
```shell
Translate the following words from English to Italain:

Input: hi 
Output: ciao

Input: bye 
Output: arrivaderci

Input: soccer 
Output: calcio

Input: word 
Output:
```

注，选择器实例化时，我们没有改变 `get_text_length` 函数实现，其默认实现为：

```python
# lengh_based.py
def _get_length_based(text: str) -> int:
    return len(re.split("\n| ", text))

...

get_text_length: Callable[[str], int] = _get_length_based
"""Function to measure prompt length. Defaults to word count."""
```


## 总结

本节课程中，我们简要介绍了LLM中的重要概念提示词`Prompt` 并学习了如何使用 `Langchain` 的重要组件提示词模板`PromptTemplate`、聊天提示词模板`ChatPromptTemplate`以及样本选择器`Example selectors`。

### 相关文档

1. [https://python.langchain.com/docs/modules/model_io/prompts/](https://python.langchain.com/docs/modules/model_io/prompts/)