from tinyrag.sentence_splitter import SentenceSplitter


def main():

    sent = """
本项目旨在构建一个小参数量的中文语言大模型，用于快速入门学习大模型相关知识，如果此项目对你有用，可以点一下start，谢谢！
模型架构：整体模型架构采用开源通用架构，包括：RMSNorm，RoPE，MHA等
实现细节：实现大模型两阶段训练及后续人类对齐，即：分词(Tokenizer) -> 预训练(PTM) -> 指令微调(SFT) -> 人类对齐(RLHF, DPO) -> 测评 -> 量化 -> 部署。
项目已部署，可以在如下网站上体验。
"""
    model_id = "models/nlp_bert_document-segmentation_chinese-base"
    ss = SentenceSplitter(use_model=True, model_path=model_id)

    result = ss.split_text(sent)

    print(len(result))
    print(result)

if __name__ == "__main__":
    main()