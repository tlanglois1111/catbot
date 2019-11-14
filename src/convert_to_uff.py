import uff

uff_model = uff.from_tensorflow_frozen_model(frozen_file='../dataset/tf/catbot-detection-graphs/catbot_detection_graph_v1.pb/frozen_inference_graph.pb',
                                             output_nodes=['detection_boxes', 'detection_classes', 'detection_scores', 'num_detections'],
                                             out_filename='../dataset/tf/catbot-detection-graphs/catbot_detection_graph_v1.pb/catbot.engine',
                                             text=True,
                                             list_nodes=False,
                                             debug_mode=False)

