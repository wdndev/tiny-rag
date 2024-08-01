
import re
from typing import List
from modelscope.pipelines import pipeline
from modelscope.pipelines import pipeline

class SentenceSplitter:
    """ 句子分割模型
        use_model: 指定是否用语义切分文档, 采取的文档语义分割模型为nlp_bert_document-segmentation_chinese-base，论文见https://arxiv.org/abs/2107.09278
    """
    def __init__(self, 
                 use_model: bool = False, 
                 model_path: str = "damo/nlp_bert_document-segmentation_chinese-base", 
                 device="cpu"
        ):
        self.use_model = use_model
        if self.use_model:
            # assert model_path == "" "模型路径为空"
            self.sent_split_pp = pipeline(
                task="document-segmentation",
                model=model_path,
                device=device
            )

    def split_text(self, sentence: str) -> List[str]:

        if self.use_model:
            # TODO: modelscope install unable to find candidates for en-core-web-sm
            result = self.sent_split_pp(documents=sentence)
            sent_list = [i for i in result["text"].split("\n\t") if i]
        else:
            sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；
            sent_list = []
            for ele in sent_sep_pattern.split(sentence):
                if sent_sep_pattern.match(ele) and sent_list:
                    sent_list[-1] += ele
                elif ele:
                    sent_list.append(ele)
        return sent_list