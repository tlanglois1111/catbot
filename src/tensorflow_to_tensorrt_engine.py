from tf_trt_models.detection import build_detection_graph
import uff


config_path = "../dataset/tf/ssd_mobilenet_v2_coco.config"
frozen_path = "../dataset/tf/catbot-detection-graphs/catbot_detection_graph_v1.pb/frozen_inference_graph.pb"
checkpoint_path = "../dataset/tf/model.ckpt-11440"
uff_model_path = "../dataset/tf/catbot-detection-graphs/catbot_detection_graph_v1.pb/catbot.engine"
output_names = ['detection_boxes', 'detection_classes', 'detection_scores', 'num_detections']
input_names = ['image_tensor']

frozen_graph, input_names, output_names = build_detection_graph(
    config=config_path,
    checkpoint=checkpoint_path,
    score_threshold=0.3,
    batch_size=1)

print("input names: {s}",input_names)
print("output names: {s}",output_names)

uff_model = uff.from_tensorflow(graphdef=frozen_graph,
                                output_names=output_names,
                                output_filename=uff_model_path,
                                text=True)
#with open(uff_model_path, "wb") as f:
#    f.write(uff_model)







