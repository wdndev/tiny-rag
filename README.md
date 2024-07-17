# Tiny RAG

## 1.简介

准备实现一个很小很小的RAG系统；

技术路线：

- 初步计划：
  - 将wiki、baike 特定的文本进行向量化后，存储在离线向量数据库中；
  - 在线检索离线数据库；
  - 将检索结果重排后，送到llm，构造prompt，llm输出结果
- 后续计划（饼）：
  - 支持从一个文件夹中向量化常见文档，如txt，doc，img等；
  - 支持调用biying的api，获取搜索结果；

拟采用技术：

- embedding：主要使用bge，同时有在线zhipu，openai embedding
- llm：主要支持qwen2本地调用，同时支持市面api
- 向量数据库：使用 faiss 向量数据库
- 检索：向量检索
- 重排
- 相似度计算：cos相似度

后续计划：

- 支持常用文档embedding
- 支持pdf，doc文档embedding
- 支持图片embedding
