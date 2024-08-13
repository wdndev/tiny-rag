# Embedding模型

## 1.选择标准

选择的标准有很多，比如模型的性能、处理速度，vector维度大小，主要从下面两个方面进行的比较：

- Huggingface趋势与下载量
- 实验对比结果

## 2 下载量

数据采集时间：2024.04.18

按趋势排行（前5名）

| Model                                                                                                                                                                               | 下载量  | 说明                                                                                                                                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [https://huggingface.co/maidalun1020/bce-embedding-base\_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1 "https://huggingface.co/maidalun1020/bce-embedding-base_v1") | 462K | 中英双语跨语言能力强。 推荐最佳实践：embedding召回top50-100片段，reranker对这50-100片段精排，最后取top5-10片段。                                                                                                                  |
| [https://huggingface.co/Salesforce/SFR-Embedding-Mistral](https://huggingface.co/Salesforce/SFR-Embedding-Mistral "https://huggingface.co/Salesforce/SFR-Embedding-Mistral")        | 90k  | 英语                                                                                                                                                                                            |
| [https://huggingface.co/aspire/acge\_text\_embedding](https://huggingface.co/aspire/acge_text_embedding "https://huggingface.co/aspire/acge_text_embedding")                        | 51K  | 中文，上升迅速, C-MTEB排行榜第一（2024.04.18）（[https://huggingface.co/spaces/mteb/leaderboard）](https://huggingface.co/spaces/mteb/leaderboard%EF%BC%89 "https://huggingface.co/spaces/mteb/leaderboard）") |
| [https://huggingface.co/jinaai/jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en "https://huggingface.co/jinaai/jina-embeddings-v2-base-en")     | 934K | 英语                                                                                                                                                                                            |
| [https://huggingface.co/jinaai/jina-embeddings-v2-base-zh](https://huggingface.co/jinaai/jina-embeddings-v2-base-zh "https://huggingface.co/jinaai/jina-embeddings-v2-base-zh")     | 5K   | 中英                                                                                                                                                                                            |

按下载量排行（最后多列出了几个中文版模型）

| Model                                                                                                                                                                                 | 下载量   | 说明                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ------------------------------- |
| [https://huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3 "https://huggingface.co/BAAI/bge-m3")                                                                         | 1964K | 多语言，bge还有英文三个版本，下载均超过1M         |
| [https://huggingface.co/BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5 "https://huggingface.co/BAAI/bge-large-zh-v1.5")                                        | 1882K | 中文                              |
| [https://huggingface.co/thenlper/gte-base](https://huggingface.co/thenlper/gte-base "https://huggingface.co/thenlper/gte-base")                                                       | 985K  | 英语                              |
| [https://huggingface.co/jinaai/jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en "https://huggingface.co/jinaai/jina-embeddings-v2-base-en")       | 934K  | 英语                              |
| [https://huggingface.co/jinaai/jina-embeddings-v2-small-en](https://huggingface.co/jinaai/jina-embeddings-v2-small-en "https://huggingface.co/jinaai/jina-embeddings-v2-small-en")    | 495K  | 英语                              |
| [https://huggingface.co/intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large "https://huggingface.co/intfloat/multilingual-e5-large")                | 816K  | 多语言                             |
| [https://huggingface.co/intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2 "https://huggingface.co/intfloat/e5-large-v2")                                              | 714K  | 英语                              |
| [https://huggingface.co/maidalun1020/bce-embedding-base\_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1 "https://huggingface.co/maidalun1020/bce-embedding-base_v1")   | 462K  | 中英双语跨语言能力强。                     |
| [https://huggingface.co/thenlper/gte-large](https://huggingface.co/thenlper/gte-large "https://huggingface.co/thenlper/gte-large")                                                    | 308K  | 英文                              |
| [https://huggingface.co/thenlper/gte-small](https://huggingface.co/thenlper/gte-small "https://huggingface.co/thenlper/gte-small")                                                    | 280K  | 英文                              |
| [https://huggingface.co/NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings "https://huggingface.co/NeuML/pubmedbert-base-embeddings")          | 184K  | 英语                              |
| [https://huggingface.co/pyannote/embedding](https://huggingface.co/pyannote/embedding "https://huggingface.co/pyannote/embedding")                                                    | 147K  | 需要注册                            |
| [https://huggingface.co/avsolatorio/GIST-large-Embedding-v0](https://huggingface.co/avsolatorio/GIST-large-Embedding-v0 "https://huggingface.co/avsolatorio/GIST-large-Embedding-v0") | 112K  | 英语，基于 BAAI/bge-large-en-v1.5 微调 |
| [https://huggingface.co/moka-ai/m3e-base](https://huggingface.co/moka-ai/m3e-base "https://huggingface.co/moka-ai/m3e-base")                                                          | 108K  | 中英，群友推荐                         |
| [https://huggingface.co/avsolatorio/GIST-Embedding-v0](https://huggingface.co/avsolatorio/GIST-Embedding-v0 "https://huggingface.co/avsolatorio/GIST-Embedding-v0")                   | 100K  | 英文                              |
| [https://huggingface.co/Salesforce/SFR-Embedding-Mistral](https://huggingface.co/Salesforce/SFR-Embedding-Mistral "https://huggingface.co/Salesforce/SFR-Embedding-Mistral")          | 91K   | 英文， 基于Mistral训练                 |
| [https://huggingface.co/aspire/acge\_text\_embedding](https://huggingface.co/aspire/acge_text_embedding "https://huggingface.co/aspire/acge_text_embedding")                          | 51K   | 中文模型 （上升非常迅速）                   |
| [https://huggingface.co/thenlper/gte-large-zh](https://huggingface.co/thenlper/gte-large-zh "https://huggingface.co/thenlper/gte-large-zh")                                           | 12K   | 中文， 入榜原因： GTE英文版下载量超大，值得关注      |
| [https://huggingface.co/jinaai/jina-embeddings-v2-base-zh](https://huggingface.co/jinaai/jina-embeddings-v2-base-zh "https://huggingface.co/jinaai/jina-embeddings-v2-base-zh")       | 5K    | 中文， 入榜原因：英文版下载量大，值得关注           |

大系列有：bge, jina, gte, bce, e5, m3e &#x20;


中文模型：bge-large-zh-v1.5, multilingual-e5-large, bce-embedding-base\_v1，m3e-base，acge\_text\_embedding

## 2.结果对比（主要参考QAnything）

- MTEB排行榜 &#x20;

  [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard "https://huggingface.co/spaces/mteb/leaderboard") &#x20;

  既包含开源，也包含API，鱼目混杂，需要进一步验证和确认。
- QAnything文档结果 &#x20;

  文档链接：[https://qanything.ai/docs/architecture](https://qanything.ai/docs/architecture "https://qanything.ai/docs/architecture")

**中英双语评估（优先Retrieval结果）：**&#x20;

| Model                  | Retrieval | Avg Score |                     |
| ---------------------- | --------- | --------- | ------------------- |
| bce-embedding-base\_v1 | 57.60     | 59.43     | zh-en， en-zh双语任务表现好 |
| multilingual-e5-large  | 56.76     | 60.50     |                     |
| bge-large-zh-v1.5      | 47.54     | 54.23     |                     |
| m3e-base               | 46.29     | 53.54     |                     |

**评测Metric：**[https://arxiv.org/pdf/2210.07316.pdf](https://arxiv.org/pdf/2210.07316.pdf "https://arxiv.org/pdf/2210.07316.pdf")

![](image/image_03V-6r-dxq.png)

**中文上Embedding模型的表现**Language: zh, Task Type: Retrieval

| Model                  | Retrieval | ReRanking |              |
| ---------------------- | --------- | --------- | ------------ |
| gte-large-zh           | 72.48     | 67.40     | 中文上表现出色      |
| bge-large-zh-v1.5      | 70.45     | 65.84     |              |
| multilingual-e5-large  | 63.65     | 57.47     | 所有任务平均上表现也很好 |
| m3e-base               | 56.91     | 59.34     |              |
| bce-embedding-base\_v1 | 53.62     | 61.67     | 单纯中文不是最好的    |

![](image/image_aNJS5vo6X7.png)

## 3.Embedding 模型推荐（中文，性能优先）

| Model                  | 下载量  | URL                                                                                                                                                                                 |
| ---------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| bce-embedding-base\_v1 | 462K | [https://huggingface.co/maidalun1020/bce-embedding-base\_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1 "https://huggingface.co/maidalun1020/bce-embedding-base_v1") |
| multilingual-e5-large  | 810K | [https://huggingface.co/intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large "https://huggingface.co/intfloat/multilingual-e5-large")              |
| gte-large-zh           | 12K  | [https://huggingface.co/thenlper/gte-large-zh](https://huggingface.co/thenlper/gte-large-zh "https://huggingface.co/thenlper/gte-large-zh")                                         |
| acge\_text\_embedding  | 51K  | [https://huggingface.co/aspire/acge\_text\_embedding](https://huggingface.co/aspire/acge_text_embedding "https://huggingface.co/aspire/acge_text_embedding")                        |

## 4.参考资料：

1. [https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83](https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83 "https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83")
2. [https://github.com/FlagOpen/FlagEmbedding/tree/master](https://github.com/FlagOpen/FlagEmbedding/tree/master "https://github.com/FlagOpen/FlagEmbedding/tree/master")
