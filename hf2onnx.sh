
hf_model_path=/public/Models/Chatglm-6B
python3 export.py --path $hf_model_path --type=chatglm-6b --export_vocab --export_test --export_path=models/onnx
python3 export.py --path $hf_model_path --type=chatglm-6b --export_embed --export_test --export_path=models/onnx
python3 export.py --path $hf_model_path --type=chatglm-6b --export_blokcs --export_test --export_path=models/onnx
python3 export.py --path $hf_model_path --type=chatglm-6b --export_lm --export_test --export_path=models/onnx
