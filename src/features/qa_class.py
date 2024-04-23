from transformers import AutoTokenizer, AutoModel, pipeline

class DocumentAssistant:
    def __init__(self, model_ckpt: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt)