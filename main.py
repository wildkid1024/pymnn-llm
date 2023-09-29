from models import Chatglm

def create_llm(model_path:str='',  backend='mnn'):
    model = None
    if "chatglm" in model_path.lower():
        model = Chatglm(backend=backend)
    return model

if __name__ == "__main__":
    onnx_model = "/public/Code/Cpp/ChatGLM-MNN/resource/models/chat_glm_onnx/"
    mnn_model = "/public/Code/Cpp/ChatGLM-MNN/resource/models/fp16"
    tokenizer_dir = "/public/Code/Cpp/mnn-llm/resource/tokenizer"
    query = "你好"
    # model = create_llm(model_path=onnx_model, backend='onnx')
    # model.load(onnx_model, tokenizer_dir)
    # model.response(query=query)

    model = create_llm(model_path=mnn_model, backend='mnn')
    model.load(mnn_model, tokenizer_dir)
    model.response(query=query)