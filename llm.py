import json
import base64
import time
import struct
import jieba
import MNN as mnn
import importlib

from engine import ORTEngine, MNNEngine 

class BaseLLM:
    def __init__(self, backend='onnx') -> None:
        self.is_single_ = False
        self.backend = backend
        self.numpy_engine = None
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

        self._executer = None

    def _tokenizer(self,):
        pass

    def _gen_attention_mask(self, ):
        pass

    def _gen_position_ids(self, ):
        pass

    def _is_stop(self, ):
        pass

    def _load_vovab(self, tokenizer_dir):
        vocab_path = tokenizer_dir + "/" + self.model_name_ + "_vocab.txt"
        print(f"load  {vocab_path}... ", end='')
        with open(vocab_path, "r", encoding='utf-8') as vocab_file:
            words = vocab_file.readlines()
            for idx, b64_word in enumerate(words):
                word = base64.b64decode(b64_word).decode()
                self._word_decoder.append(word)
                self._word_encoder[word] = idx
            print("Done!")

    def load(self, model_dir, tokenizer_dir):
        self._model_dir = model_dir
        self._tokenizer_dir = tokenizer_dir

        # 1. load vocab
        self._load_vovab(tokenizer_dir=tokenizer_dir)

        if self.backend == "onnx":
            self._executer = ORTEngine(layer_nums=self.layer_nums_, past_kv_shape=self.key_value_shape_)
            self.numpy_engine = importlib.import_module("numpy")
        elif self.backend == "mnn":
            self._executer = MNNEngine(layer_nums=self.layer_nums_, past_kv_shape=self.key_value_shape_)
            self.numpy_engine = importlib.import_module("MNN.numpy")

        self._executer._load_model(model_dir)
    
    def gen_embedding(self, input_ids):
        seq_len = len(input_ids)
        embedding = self.numpy_engine.empty([seq_len, 1, self.hidden_], dtype=self.numpy_engine.float32)
        # embedding = _Input({1, static_cast<int>(seq_len), hidden_}, NCHW)
        size = self.hidden_ * 4
        file_path = self._model_dir + "/slim_word_embeddings.bin"
        file_path = "/public/Code/Cpp/ChatGLM-MNN/resource/models/int4/slim_word_embeddings.bin"
        with open(file_path, "rb") as f:
            for i in range(seq_len):
                f.seek(input_ids[i]*size)
                buffer = f.read(size)
                buffer = struct.unpack('f'*self.hidden_, buffer)
                for j in range(self.hidden_):
                    # b = buffer[j*4: (j*4)+4]
                    embedding[i][0][j] = float(buffer[j])
        return embedding

    def tokenizer_encode(self, input_str):
        ids = []
        dict_path = self._tokenizer_dir + "/jieba.dict.utf8"
        model_path = self._tokenizer_dir + "/hmm_model.utf8"
        user_dict_path = self._tokenizer_dir + "/user.dict.utf8"
        idf_path = self._tokenizer_dir + "/idf.utf8"
        stopWord_path = self._tokenizer_dir + "/stop_words.utf8"
        jieba.load_userdict(user_dict_path)
        # jieba.set_dictionary(dict_path)
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
        attention_mask = self._gen_attention_mask(seq_len)
        position_ids = self._gen_position_ids(seq_len)
        id = self._executer.forward(input_ids, attention_mask, position_ids)
        return id

    def response(self, query):
        self._executer.reset_kv()
        
        # response
        st = time.time()
        input_ids = self._tokenizer(query, use_hf=True)
        print(input_ids)
        token = self.forward(input_ids)
        output_str = self.decode(token)
        print(output_str, end='', flush=True)

        while self.gen_seq_len_ < 2048:
            token = self.forward([token])
            self.gen_seq_len_ += 1
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




