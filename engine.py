import MNN as mnn
from typing import List
import os
import json

class BaseEngine:
    def __init__(self, layer_nums=0, past_kv_shape=None, **kwargs) -> None:
        self.layer_nums_ = layer_nums
        self.key_value_shape_ = past_kv_shape

        self.all_seq_len_ = 0
        self.gen_seq_len_ = 0

        self._modules = []
        self._past_key_values = []
        self.model_dir = ""
        self.tokenizer_dir = ""
    
    @property
    def _is_single(self, ):
        return os.path.isfile(self.model_dir)
    
    def reset_kv(self, ):
        pass

    def _load_model(self, ):
        pass

    def forward(self, ):
        pass

class ORTEngine(BaseEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _load_model(self, model_dir):
        import onnxruntime as ort
        self._model_dir = model_dir

        load_progress_ = 0.0
        if self._is_single:
            self._modules.append(ort.InferenceSession(model_dir, providers=['CPUExecutionProvider']))
        else:
            self._modules = [None] * (self.layer_nums_ + 2)
            step = 100.0 / (self.layer_nums_ + 2)

            # load lm model
            lm_model_path = model_dir + "/lm.onnx"
            embedding_model_path = model_dir + "/embedding2.onnx"
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
                model_path = model_dir + f"/glm_block_{i}.onnx"
                print("[%3.0f%% ] load %s model ... "%(load_progress_, model_path), end='')
                self._modules[i] = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                print("Done!", flush=True)
                # if i == 1: break

    def forward(self, input_ids, attention_mask, position_ids)->List[int]:
        import numpy as np
        inputs_ids_ = np.array(input_ids, dtype=np.int64)
        print(input_ids)
        id = -1

        attention_mask = attention_mask > 0
        position_ids = position_ids.astype(np.int64)
        
        if self._is_single:
            # single model
            outputs = self._modules[-1].run([inputs_ids_, attention_mask, position_ids, self._past_key_values[0]])
            id = outputs[0].read()[0]
            self._past_key_values[0] = outputs[1]
        else:
            # split block models
            hidden_states = self._modules[self.layer_nums_ + 1].run(input_feed={"input_ids": inputs_ids_}, output_names=None)[0]
           
            # hidden_states = self.gen_embedding(inputs_ids_)
            # hidden_states = np.random.random([3, 1, 4096])
            # print("hidden_states shape:", hidden_states.shape)
            # print("attention_mask shape:", attention_mask.shape)
            # print("position_ids shape:", position_ids.shape)
            # print("_past_key_values shape:", self._past_key_values[0].shape)

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
                # if i == 1: break

            outputs = self._modules[self.layer_nums_].run(input_feed={"hidden_states": hidden_states[-1]}, output_names=None)
            id = outputs[0]
        return id
    
    def reset_kv(self, ):
        import numpy as np
        if self._is_single:
            self.key_value_shape_.insert(0, self.layer_nums_)
            self._past_key_values.append(np.empty(self.key_value_shape_, dtype=np.float32))
        else:
            for i in range(self.layer_nums_):
                self._past_key_values.append(np.empty(self.key_value_shape_, dtype=np.float32))

class MNNEngine(BaseEngine):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def _load_model(self, model_dir):
        # return super()._load_model()
        self._model_dir = model_dir
        config = {
            "backend": "CPU",
            "precision": "low",
            # "numThread": 4,
            #   "saveTensors": "",
            #   "inputPaths":"",
            #   "outputPaths": "",
        }
        self._runtime_manager = mnn.nn.create_runtime_manager(json.dumps({}))

        load_progress_ = 0.0
        module_config = {
            "shape_mutable": True,
            "rearrange": True,
            "backend": mnn.expr.Backend.CPU,
            "precision_mode": mnn.expr.PrecisionMode.Low,
            # "runtime_manager": self._runtime_manager
        }
        if self._is_single:
            model_path = self.model_dir_
            external_path = self.model_dir_ + ".weight"
            print("load {model_path} ... ", end='')
            self._runtime_manager.set_external(external_path)
            module_config['runtime_manager'] = self._runtime_manager
            self._modules = mnn.nn.load_module_from_file(model_path, 
                                                         ("input_ids", "attention_mask", "position_ids", "past_key_values"), 
                                                         ("token_id", "presents"), **module_config)
            print("Done!", flush=True)
        else:
            # 2. load models
            self._modules = [None] * (self.layer_nums_ + 2)
            step = 100.0 / (self.layer_nums_ + 2)

            # load lm model
            lm_model_path = model_dir + "/lm.mnn"
            embedding_model_path = model_dir + "/embedding.mnn"
            print("[%3.0f%% ] load %s model ... "%(load_progress_, lm_model_path), end='')
            self._modules[self.layer_nums_] = mnn.nn.load_module_from_file(lm_model_path, [], [], **module_config)
            print("Done!")
            load_progress_ += step
            print("[%3.0f%% ] load %s model ... "%(load_progress_, embedding_model_path), end='', flush=True)
            self._modules[self.layer_nums_ + 1] = mnn.nn.load_module_from_file(embedding_model_path, [], [], **module_config)
            print("Done!")
            load_progress_ += step

            # load glm_block models
            for i in range(self.layer_nums_):
                load_progress_ += step
                model_path = model_dir + f"/glm_block_{i}.mnn"
                print("[%3.0f%% ] load %s model ... "%(load_progress_, model_path), end='')
                self._modules[i] = mnn.nn.load_module_from_file(
                    model_path,
                    ("inputs_embeds", "attention_mask", "position_ids", "past_key_values"),
                    ("hidden_states", "presents"), **module_config)
                print("Done!", flush=True)
                # if i == 1: break
    
    def forward(self, input_ids, attention_mask, position_ids):
        import MNN.numpy as np
        seq_len = len(input_ids)
        inputs_ids_ = np.empty([seq_len, ], np.int64)
        for i in range(seq_len): inputs_ids_[i] = int(input_ids[i]) 
        # attention_mask = self._gen_attention_mask(seq_len)
        # position_ids = self._gen_position_ids(seq_len)
        id = -1
        if self._is_single:
            # single model
            outputs = self._modules[-1].onForward([inputs_ids_, attention_mask, position_ids, self._past_key_values[0]])
            id = outputs[0].read()[0]
            self._past_key_values[0] = outputs[1]
        else:
            # split block models
            hidden_states = self._modules[self.layer_nums_ + 1].onForward([inputs_ids_])[0]
            # hidden_states = self.gen_embedding(inputs_ids_)
            # hidden_states = np.random.random([4, 1, 4096])
            # attention_mask = np.random.randint(0, 1, [1, 1, 4, 4])
            # position_ids = np.random.randint(0, 4, [1, 2, 4])
            # self._past_key_values = np.random.random([2, 0, 1, 32, 128])
            # print("hidden_states shape:", hidden_states.shape)
            # print("attention_mask shape:", attention_mask.shape)
            # print("position_ids shape:", position_ids.shape)
            # print("_past_key_values shape:", self._past_key_values[0].shape)

            for i in range(self.layer_nums_):
                outputs = self._modules[i].onForward([hidden_states, attention_mask, position_ids, self._past_key_values[i]])
                # print(outputs)
                hidden_states = outputs[0]
                self._past_key_values[i] = outputs[1]
                # if i == 1: break
                
            # print("last layer:", hidden_states)
            logisits = self._modules[self.layer_nums_].onForward([hidden_states[-1]])
            id = logisits[0].read()
            print(id)
        self.all_seq_len_ += seq_len
        self.gen_seq_len_+= 1
        return id
    
    def reset_kv(self, ):
        import MNN.numpy as np
        if self._is_single:
            self.key_value_shape_.insert(0, self.layer_nums_)
            self._past_key_values.append(np.empty(self.key_value_shape_, dtype=np.float32))
        else:
            for i in range(self.layer_nums_):
                self._past_key_values.append(np.empty(self.key_value_shape_, dtype=np.float32))