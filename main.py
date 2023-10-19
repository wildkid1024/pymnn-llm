from transformers import AutoTokenizer
from models import (Chatglm, Chatglm2, Qwen, Baichuan2)
from models import ModelType
from engine import EngineType
from tokenizer import Tokenizer, ChatglmTokenizer

def create_llm(model_path:str="", vocab_path:str="", model_type=ModelType.Chatglm, backend=EngineType.MNN):
    model = None
    tokenizer = Tokenizer(vocab_path)
    if model_type == ModelType.Chatglm:
        model = Chatglm(backend=backend)
        tokenizer = ChatglmTokenizer(vocab_path)
    elif model_type == ModelType.Chatglm2:
        model = Chatglm2(backend=backend)
    elif model_type == ModelType.Qwen:
        model = Qwen(backend=backend)
    elif model_type == ModelType.Baichuan2:
        model = Baichuan2(backend=backend)
    else:
        raise NotImplementedError(f"unsupport model type!")
    model.load(model_dir=model_path)    
    return model, tokenizer

def chat():
    onnx_model = "models/onnx"
    mnn_model = "models/mnn/"
    query = "你好"
    tokenizer = AutoTokenizer.from_pretrained("/home/pan/Public/Models/models-hf/chatglm-6b", trust_remote_code=True)
    vocab_path = onnx_model + "/Chatglm_6b_vocab.txt"

    model, tokenizer = create_llm(model_path=mnn_model, vocab_path=vocab_path, backend=EngineType.MNN)
    model.response(query=query, tokenizer=tokenizer)

if __name__ == "__main__":
    chat()