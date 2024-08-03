# Tiny RAG

## 1.简介

准备实现一个很小很小的RAG系统；

技术路线：

- 初步计划：
  - 将wiki、baike 特定的文本进行向量化后，存储在离线向量数据库中；
  - 在线检索离线数据库；
  - 将检索结果重排后，送到llm，构造prompt，llm输出结果

拟采用技术：

- 文档解析：支持 txt, markdown, pdf, word, ppt，图像等向量化
- 文档embedding：主要使用bge，同时有在线zhipu，openai embedding
- 图像embedding: 使用 clip-vit 进行图像embedding
- 句子切分：支持模型和规则切分
  - 模型切分： 采取的文档语义分割模型为 nlp_bert_document-segmentation_chinese-base， 论文见https://arxiv.org/abs/2107.09278
  - 规则切分
- llm：主要支持qwen2本地调用，同时支持市面api
- 向量数据库： 使用 faiss 向量数据库
- 召回：实现多路召回
  - 向量召回：将句子 bge 模型编码后，使用 faiss 数据库召回 top-n
  - bm25召回：使用 bm25 算法关键字召回
- 重排： 结合多路召回的结果，使用 bge-ranker 模型进行重排

## 2.项目文件简介

具体如下所示：

```shell
├─data                  # 存放原始数据和向量数据库文件夹
│  ├─db                 ## 缓存数据库文件 
│  │  ├─bm_corpus       ### bm25 召回缓存数据
│  │  └─faiss_idx       ### 向量召回缓存数据
│  ├─parser_test
│  └─raw_data           ## 原始数据
├─doc                   # 相关文档
├─models                # 模型存放文件夹
│  ├─bge-reranker-base
│  ├─bge-small-zh-v1.5
│  ├─clip-ViT-B-32
│  └─nlp_bert_document-segmentation_chinese-base
├─script                # 执行脚本文件
├─test                  # 测试文件
└─tinyrag               # tinyrag
    ├─embedding         ## embedding
    ├─llm               ## llm
    ├─parser            ## 文档解析
    └─searcher          ## 根据query搜索相关文档
       ├─bm25_recall    ### bm25召回
       ├─emb_recall     ### 向量召回
       └─reranker       ### 重排

```


