# pymnn-llm

pymnn-llm项目是mnn-llm(ChatGLM-MNN)项目的python版本，相比与cpp版本，增加了onnx运行后端。
当前项目是一个假期toy项目，只保证了模型的正常运行，但不一定保证模型得到正确的预期结果。

## 运行步骤

1. 确保已经编译安装了pymnn，详细请参考mnn项目下的python api的安装说明
2. 导出hf模型到onnx，可以使用export.py的脚本进行安装，这里参考了llm-export项目中的做法
3. 使用mnnconvert工具将onnx模型转化为mnn模型，具体用法可以参考mnn。
4. 运行main程序，传入对应的模型路径，选择对应的后端，得到输出结果


## 扩展性

### 前端模型扩展

前台模型扩展需要在models.py下添加对应的模型，继承Basellm基类，
并实现：
_tokenizer,

_gen_attention_mask,

_gen_position_ids,

_is_stop,

等类成员方法。


### 后端engine扩展

后端运行时扩展需要在engine.py下添加对应的推理运行时，继承BaseEngine基类,
并实现：

reset_kv,

_load_model,

forward,

等类成员方法。


## FAQ

-Q: 出现错误Acquire buffer size

爆内存了，检查输入的数据量是否过大，输入的shape是否正确以及输入数据类型是否一致。



## 参考

1. mnn: https://github.com/alibaba/MNN
2. mnn-llm: https://github.com/wangzhaode/mnn-llm
3. llm-export: https://github.com/wangzhaode/llm-export
