import tensorrt as trt
import sys
import uff

uff_model = uff.from_tensorflow_frozen_model('../dataset/tf/trained-inference-graphs/trained_inference_graph_v1.pb/frozen_inference_graph.pb', ['../dataset/tf'], out_filename='model.uff')

