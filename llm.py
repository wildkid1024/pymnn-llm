import time
import struct
import MNN as mnn
import importlib
from typing import List

from engine import ORTEngine, MNNEngine 
from engine import EngineType
from tokenizer import Tokenizer

class BaseLLM:
    def __init__(self, backend=EngineType.MNN) -> None:
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

        self._executer = None
    
    def _build_inputs(self, ):
        raise NotImplementedError
    
    def _gen_attention_mask(self, ):
        raise NotImplementedError

    def _gen_position_ids(self, ):
        raise NotImplementedError

    def _is_stop(self, ):
        raise NotImplementedError

    def load(self, model_dir):
        self._model_dir = model_dir
        
        #  load model
        if self.backend == EngineType.ORT:
            self._executer = ORTEngine(layer_nums=self.layer_nums_, past_kv_shape=self.key_value_shape_)
            self.numpy_engine = importlib.import_module("numpy")
        elif self.backend == EngineType.MNN:
            self._executer = MNNEngine(layer_nums=self.layer_nums_, past_kv_shape=self.key_value_shape_)
            self.numpy_engine = importlib.import_module("MNN.numpy")
        else:
            raise NotImplementedError(f"unsupport backend type!")

        self._executer._load_model(model_dir)
    
    def gen_embedding(self, input_ids):
        seq_len = len(input_ids)
        embedding = self.numpy_engine.empty([seq_len, 1, self.hidden_], dtype=self.numpy_engine.float32)
        size = self.hidden_ * 4
        file_path = self._model_dir + "/slim_word_embeddings.bin"
        with open(file_path, "rb") as f:
            for i in range(seq_len):
                f.seek(input_ids[i]*size)
                buffer = f.read(size)
                buffer = struct.unpack('f'*self.hidden_, buffer)
                for j in range(self.hidden_):
                    # b = buffer[j*4: (j*4)+4]
                    embedding[i][0][j] = float(buffer[j])
        return embedding
    

    def forward(self, input_ids:List[int])->int:
        seq_len = len(input_ids)
        attention_mask = self._gen_attention_mask(seq_len)
        position_ids = self._gen_position_ids(seq_len)
        token_id = self._executer.forward(input_ids, attention_mask, position_ids)
        return token_id

    def response(self, query, tokenizer:Tokenizer=None, max_seq_len:int=2048):
        self._executer.reset_kv()
        
        # response
        st = time.time()
        input_ids = tokenizer.encode(query)
        self._context_len_ = len(input_ids)

        print(input_ids)
        token = self.forward(input_ids)
        output_str = tokenizer.decode([token])
        print(output_str, end='', flush=True)

        while self.gen_seq_len_ < max_seq_len:
            token = self.forward([token])
            self.gen_seq_len_ += 1
            if self._is_stop(token):
                print(flush=True)
                break

            word = tokenizer.decode([token])
            print(output_str, end='', flush=True)
            output_str += word

            et = time.time()
            duration = et - st
            print("\n[speed: {} tok/s]\n".format(self.gen_seq_len_ / (duration)))
        return output_str




