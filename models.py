import numpy as np
from llm import BaseLLM
from enum import Enum

class ModelType(Enum):
    Chatglm = 0 
    Chatglm2 = 1
    Qwen = 128
    Baichuan2 = 256


# chatglm-6b
class Chatglm(BaseLLM):
    def __init__(self, backend=None) -> None:
        super().__init__(backend=backend)
        self.model_name_ = f"{ModelType.Chatglm.name}_6b" 
        self.layer_nums_ = 28
        self.key_value_shape_ = [2, 0, 1, 32, 128]

        self._context_len_ = 0

    def _gen_attention_mask(self, seq_len:int):
        attention_mask = np.zeros([1, 1, seq_len, seq_len], dtype=np.int32)
        if seq_len > 1:
            for i in range(seq_len-1):
                attention_mask[0][0][i][-1] = 1
        return attention_mask

    def _gen_position_ids(self, seq_len:int):
        position_ids = np.zeros([1, 2, seq_len], dtype=np.int32)
        if seq_len == 1:
            position_ids[0][0][0] = 1
            position_ids[0][1][0] = self.all_seq_len_ - (self._context_len_ - 2) 
        else:
            for i in range(seq_len):
                position_ids[0][0][i] = i
                position_ids[0][1][i] = 0
    
            position_ids[0][1][seq_len - 1] = 1
        return position_ids

    def _is_stop(self, token_id:int)->bool:
        return token_id == 130005

# chatglm2-6b
class Chatglm2(BaseLLM):
    def __init__(self, backend='onnx') -> None:
        super().__init__(backend)
        self.model_name_ = ModelType.Chatglm2.name
        self.layer_nums_ = 28
        self.key_value_shape_ = [2, 0, 1, 32, 128]

        self._context_len_ = 0

    def _tokenizer(self, query:str, use_hf=False):
        prompt = "\n问：\n" + query + "答：\n"
        ids = self.tokenizer_encode(prompt)
        ids.insert(0, 64792)
        ids.insert(0, 64790)
        self._context_len_ = len(ids)
        return ids

    def _gen_attention_mask(self, seq_len:int):
        attention_mask = self.numpy_engine.zeros([1, 1, seq_len, seq_len], dtype=self.numpy_engine.int32, order='C')
        if seq_len > 1:
            for i in range(seq_len):
                for j in range(seq_len):
                    attention_mask[0][0][i][j] = int(j>i) 
        return attention_mask

    def _gen_position_ids(self, seq_len:int):
        position_ids = self.numpy_engine.zeros([seq_len], dtype=self.numpy_engine.int32, order='C')
        if seq_len == 1:
            position_ids[0] = self.gen_seq_len_
        else:
            for i in range(seq_len):
                position_ids[i] = i
        return position_ids

    def _is_stop(self, token_id:int)->bool:
        return token_id <= 2
    
# Qwen-7B
class Qwen(BaseLLM):
    def __init__(self, backend='onnx') -> None:
        super().__init__(backend)

        self.model_name_ = ModelType.Qwen.name
        self.layer_nums_ = 32
        self.key_value_shape_ = [2, 0, 1, 32, 128]

        self._context_len_ = 0

    def _gen_attention_mask(self, seq_len:int):
        attention_mask = self.numpy_engine.empty([1, 1, seq_len, seq_len], dtype=self.numpy_engine.int32)
        if seq_len > 1:
            for i in range(seq_len):
                for j in range(seq_len):
                    attention_mask[0][0][i][j] = int(j<=i) 
        return attention_mask

    def _gen_position_ids(self, seq_len:int):
        position_ids = self.numpy_engine.empty([seq_len], dtype=self.numpy_engine.int32)
        if seq_len == 1:
            position_ids[0] = self.all_seq_len_
        else:
            for i in range(seq_len):
                position_ids[i] = i
        return position_ids

    def _is_stop(self, token_id:int)->bool:
        return token_id >= 151645

# Baichuan2_7b
class Baichuan2(BaseLLM):
    def __init__(self, backend='onnx') -> None:
        super().__init__(backend)
        self.model_name_ = ModelType.Baichuan2.name
        self.layer_nums_ = 32
        self.key_value_shape_ = [2, 1, 32, 0, 128];

        self._context_len_ = 0

    def _gen_attention_mask(self, seq_len:int):
        attention_mask = self.numpy_engine.zeros([1, 1, seq_len, seq_len], dtype=self.numpy_engine.float32)
        if seq_len > 1:
            for i in range(seq_len):
                for j in range(seq_len):
                    attention_mask[0][0][i][j] = int(j>=i) * 1e-9
        else:
            attention_mask = self.numpy_engine.zeros([1, 1, seq_len, seq_len+1], dtype=self.numpy_engine.float32)
        return attention_mask

    def _gen_position_ids(self, seq_len:int):
        position_ids = self.numpy_engine.zeros([1, seq_len], dtype=self.numpy_engine.int32)
        if seq_len == 1:
            position_ids[0][0] = self.all_seq_len_
        else:
            for i in range(seq_len):
                position_ids[0][i] = i
        return position_ids

    def _is_stop(self, token_id:int)->bool:
        return token_id == 2 
    