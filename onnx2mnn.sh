
mnnconvert -f ONNX --modelFile ./models/onnx/embedding.onnx --MNNModel ./models/mnn/embedding.mnn
for i in `seq 0 27`
do
    src=./models/onnx/block_$i.onnx
    dst=./models/mnn/block_$i.mnn
    # mnnconvert -f ONNX --modelFile $src --MNNModel $dst --bizCode mnn --weightQuantBits 4 --weightQuantAsymmetric
    mnnconvert -f ONNX --modelFile $src --MNNModel $dst --fp16
done
mnnconvert -f ONNX --modelFile ./models/onnx/lm.onnx --MNNModel ./models/mnn/lm.mnn