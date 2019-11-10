from tf_trt_models.detection import download_detection_model, build_detection_graph
from tensorflow.python.compiler.tensorrt import trt_convert as trt

config_path = "../dataset/tf/ssd_mobilenet_v2_coco.config"
checkpoint_path = "../dataset/tf/model.ckpt-7849"
uff_model_path = "../dataset/tf/catbot-detection-graphs/catbot_detection_graph_v1.pb/frozen_inference_graph.uff"
engine_path = "../dataset/tf/catbot-detection-graphs/catbot_detection_graph_v1.pb/catbot.engine"

frozen_graph, input_names, output_names = build_detection_graph(
    config=config_path,
    checkpoint=checkpoint_path,
    score_threshold=0.3,
    batch_size=1
)

print(output_names)

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

with open(engine_path, "wb") as f:
    f.write(trt_graph.SerializeToString())
