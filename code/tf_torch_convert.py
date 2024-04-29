import tf2onnx
import onnx
from onnx2pytorch import ConvertModel

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

loaded_model_SqueezeformerM = tf.keras.models.load_model("/home/ubuntu/asr/models/squeezeformer-M.h5")
onnx_model_SqueezeformerM, _ = tf2onnx.convert.from_keras(loaded_model_SqueezeformerM)
pytorch_model_SqueezeformerM = ConvertModel(onnx_model_SqueezeformerM)
torch.save(pytorch_model_SqueezeformerM.state_dict(),"/home/ubuntu/asr/models/pytorch_squeezeformer-M.pth" )