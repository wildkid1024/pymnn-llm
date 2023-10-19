import time
import copy
import base64
import jieba
import struct

import onnx
import onnxruntime as ort

import numpy as np
import MNN.numpy as _np 

class LLM:
    def __init__(self, tokenizer=None, ) -> None:
        self.is_single_ = False
        self.layer_nums_ = 0
        self.hidden_ = 4096
        self.key_value_shape_ = []
        self.model_name_ = ""

        self.gen_seq_len_ = 0
        self.all_seq_len_ = 0
        self.load_progress_ = 0.0

        self._runtime_manager = None
        self._modules = []
        self._past_key_values = []

        self._model_dir = ""
        self._tokenizer_dir = ""

        self._word_encoder = {}
        self._word_decoder = []
    
    def _tokenizer(self, ):
        pass

    def _gen_attention_mask(self, ):
        pass

    def _gen_position_ids(self, ):
        pass

    def _is_stop(self, ):
        pass

    def load(self, model_dir, tokenizer_dir):
        self._model_dir = model_dir
        self._tokenizer_dir = tokenizer_dir

        # 1. load vocab
        vocab_path = tokenizer_dir + "/" + self.model_name_ + "_vocab.txt"
        print(f"load  {vocab_path}... ", end='')
        with open(vocab_path, "r", encoding='utf-8') as vocab_file:
            words = vocab_file.readlines()
            for idx, b64_word in enumerate(words):
                word = base64.b64decode(b64_word).decode()
                self._word_decoder.append(word)
                self._word_encoder[word] = idx
            print("Done!")

        load_progress_ = 0.0
        if self.is_single_:
            pass
        else:
            # 2. load models
            self._modules = [None] * (self.layer_nums_ + 2)
            step = 100.0 / (self.layer_nums_ + 2)

            # load lm model
            lm_model_path = model_dir + "/lm.onnx"
            embedding_model_path = model_dir + "/embedding.onnx"
            load_progress_ += step
            print("[%3.0f%% ] load %s model ... "%(load_progress_, lm_model_path), end='')
            self._modules[self.layer_nums_] = ort.InferenceSession(lm_model_path, providers=['CPUExecutionProvider'])
            print("Done!")
            load_progress_ += step
            print("[%3.0f%% ] load %s model ... "%(load_progress_, embedding_model_path), end='', flush=True)
            self._modules[self.layer_nums_ + 1] = ort.InferenceSession(embedding_model_path, providers=['CPUExecutionProvider'])
            print("Done!")

            # load glm_block models
            for i in range(self.layer_nums_):
                load_progress_ += step
                model_path = model_dir + f"/block_{i}.onnx"
                print("[%3.0f%% ] load %s model ... "%(load_progress_, model_path), end='')
                if i < 7:
                    self._modules[i] = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                else:
                    self._modules[i] = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                # if i == 1: break 
                print("Done!", flush=True)
            
    def gen_embedding(self, input_ids):
        seq_len = len(input_ids)
        embedding = np.empty([seq_len, 1, self.hidden_], dtype=np.float32)
        # embedding = _Input({1, static_cast<int>(seq_len), hidden_}, NCHW)
        size = self.hidden_ * 4
        file_path = self._model_dir + "/slim_word_embeddings.bin"
        with open(file_path, "rb") as f:
            for i in range(seq_len):
                f.seek(input_ids[i]*size)
                buffer = f.read(size)
                buffer = struct.unpack('f'*self.hidden_, buffer)
                for j in range(self.hidden_):
                    embedding[i][0][j] = float(buffer[j])
        return embedding

    def tokenizer_encode(self, input_str):
        ids = []
        # user_dict_path = self._tokenizer_dir + "/user.dict.utf8"
        # jieba.load_userdict(user_dict_path)

        words = jieba.cut(input_str, HMM=True)
        for word in words:
            id = self._word_encoder.get(word, -1)
            if id >= 0: ids.append(id)
        return ids

    def decode(self, id):
        word = self._word_decoder[id]
        # Fix utf-8 garbled characters
        if len(word) == 6 and word.startswith('<0x'):
            word = word[1:-1].decode()
        return word

    def load_progress(self, ):
        pass

    def forward(self, input_ids):
        seq_len = len(input_ids)
        inputs_ids_ = np.array(input_ids)
        attention_mask = self._gen_attention_mask(seq_len)
        position_ids = self._gen_position_ids(seq_len)
        id = -1
        if self.is_single_:
            # single model
            outputs = self._modules[-1].run([inputs_ids_, attention_mask, position_ids, self._past_key_values[0]])
            id = outputs[0].read()[0]
            self._past_key_values[0] = outputs[1]
        else:
            # split block models
            hidden_states = self._modules[self.layer_nums_ + 1].run(input_feed={"input_ids": inputs_ids_}, output_names=None)[0]

            for i in range(self.layer_nums_):
                input_feed = {
                    'inputs_embeds': hidden_states,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids,
                    'past_key_values': self._past_key_values[i], 
                }
                outputs = self._modules[i].run(input_feed=input_feed, output_names=None)
                hidden_states = outputs[0]
                self._past_key_values[i] = outputs[1]
            
            outputs = self._modules[self.layer_nums_].run(input_feed={"hidden_states": hidden_states}, output_names=None)
            id = outputs[0]
        self.all_seq_len_ += seq_len
        self.gen_seq_len_+= 1
        return id

    def response(self, query):
        if self.is_single_:
            self.key_value_shape_.insert(0, self.layer_nums_)
            self._past_key_values.append(np.empty(self.key_value_shape_))
        else:
            for i in range(self.layer_nums_):
                self._past_key_values.append(np.empty(self.key_value_shape_, dtype=np.float32))

        # response
        st = time.time()
        input_ids = self._tokenizer(query)
        print("jieba decode:", input_ids)
        token = self.forward(input_ids)
        output_str = self.decode(token)
        print(output_str, end='', flush=True)

        while self.gen_seq_len_ < 2048:
            token = self.forward([token])
            if self._is_stop(token):
                    print(flush=True)
                    break

            word = self.decode(token)
            print(output_str, end='', flush=True)
            output_str += word

            et = time.time()
            duration = et - st
            print("\n[speed: {} tok/s]\n".format(self.gen_seq_len_ / (duration)))
        return output_str

    def reset(self, ):
        pass

class Chatglm(LLM):
    def __init__(self, tokenizer=None) -> None:
        super().__init__(tokenizer)
        self.model_name_ = "Chatglm_6b"
        self.layer_nums_ = 28
        self.key_value_shape_ = [2, 0, 1, 32, 128]

        self._context_len_ = 0
        
    def _tokenizer(self, query:str):
        ids = self.tokenizer_encode(query)
        ids.insert(0, 5)
        self._context_len_ = len(ids)
        ids.append(130001)
        ids.append(130004)
        # from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("/public/Models/chatglm-6b", trust_remote_code=True)
        # input_ids = tokenizer(query)
        # print("tokenizer decode:", input_ids)
        # ids = input_ids['input_ids']
        return ids

    def _gen_attention_mask(self, seq_len:int):
        attention_mask = np.empty([1, 1, seq_len, seq_len], dtype=np.bool_)
        if seq_len > 1:
            for i in range(seq_len-1):
                attention_mask[0][0][i][-1] = 1
        return attention_mask

    def _gen_position_ids(self, seq_len:int):
        position_ids = np.empty([1, 2, seq_len], dtype=np.int64)
        if seq_len == 1:
            position_ids[0][0][0] = 1
            position_ids[0][1][0] = self.all_seq_len_ - self._context_len_
        else:
            for i in range(seq_len):
                position_ids[0][0][i] = i
                position_ids[0][1][i] = 0
    
            position_ids[0][1][seq_len - 1] = 1
        return position_ids

    def _is_stop(self, token_id:int)->bool:
        return token_id == 130005

def create_llm(mnn_path:str):
    single_file = mnn_path.endswith(".onnx")
    model = None
    # if "chatglm" in mnn_path.lower():
    model = Chatglm()
    model.is_single_ = single_file
    return model

def infer_block():
    hidden_states = np.random.random([3, 1, 4096]).astype(np.float32)
    attention_mask = np.random.randint(low=0, high=1, size=[1, 1, 3, 3]).astype(np.bool_)
    position_ids = np.random.randint(low=0, high=3, size=[1, 2, 3])
    past_key_values = np.random.random([2, 0, 1, 32, 128]).astype(np.float32)
    input_feed = {
        "inputs_embeds": hidden_states,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values
    }
    onnx_path = "/public/Code/Cpp/ChatGLM-MNN/resource/models/chat_glm_onnx/glm_block_0.onnx"
    onnx_model = onnx.load(onnx_path)
    ori_output = copy.deepcopy(onnx_model.graph.output)
    for node in onnx_model.graph.node:
        for output in node.output:
            if output == "onnx::Transpose_693":
                onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    session = ort.InferenceSession(onnx_model.SerializeToString(), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    outputs = session.run(output_names=None, input_feed=input_feed)
    for o in outputs: print(o.shape)
    print(outputs)

def infer_lm():
    hidden_states = np.random.random([1, 4096]).astype(np.float32)
    input_feed = {
        "hidden_states": hidden_states
    }
    onnx_path = "/public/Code/Cpp/ChatGLM-MNN/resource/models/chat_glm_onnx/lm.onnx"
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    inputs_name = [(x.name, x.shape) for x in session.get_inputs()]
    outputs_name = [(x.name, x.shape) for x in session.get_outputs()]
    print(inputs_name)
    print(outputs_name)
    outputs = session.run(output_names=["token_id"], input_feed=input_feed)
    print(outputs)


if __name__ == "__main__":
    # infer_block()
    # infer_lm()
    onnx_model = "./onnx"
    query = "飞机为什么能够飞行"
    model = create_llm(mnn_path=onnx_model)
    model.load(onnx_model, onnx_model)
    model.response(query=query)