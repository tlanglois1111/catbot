import tensorflow as tf

FILE = '../dataset/tf/catbot-detection-graphs/catbot_detection_graph_v1.pb/frozen_inference_graph.pb'

gf = tf.GraphDef()
gf.ParseFromString(open('../dataset/tf/catbot-detection-graphs/catbot_detection_graph_v1.pb/frozen_inference_graph.pb', 'rb').read())

tf_node_list = [n.name for n in gf.node]

for m in tf_node_list:
    print(m)

