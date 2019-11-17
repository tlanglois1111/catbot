import uff
import tensorrt as trt

config_path = "../dataset/tf/ssd_mobilenet_v2_coco.config"
frozen_path = "../dataset/tf/catbot-detection-graphs/catbot_detection_graph_v1.pb/frozen_inference_graph.pb"
checkpoint_path = "../dataset/tf/model.ckpt-28553"
uff_model_path = "../dataset/tf/catbot-detection-graphs/catbot_detection_graph_v1.pb/catbot.uff"
tensorrt_model_path = "../dataset/tf/catbot-detection-graphs/catbot_detection_graph_v1.pb/catbot.engine"
output_names = ['detection_boxes', 'detection_classes', 'detection_scores', 'num_detections']
input_names = ['image_tensor']


def remove_asserts(graph):
    all_assert_nodes = graph.find_nodes_by_op("Assert")
    graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)

    return graph


uff_model = uff.from_tensorflow_frozen_model(frozen_file=frozen_path,
                                             output_filename=uff_model_path,
                                             preprocessor="remove_asserts.py",
                                             text=True)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

trt.init_libnvinfer_plugins(TRT_LOGGER, "")
with trt.Builder(TRT_LOGGER) as builder:
    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 28
    builder.fp16_mode = True
    network = builder.create_network()
    print("network created")
    parser = trt.UffParser()
    parser.register_input(input_names[0], [3, 300, 300])
    for output_node in output_names:
        parser.register_output(output_node)
    print("parsing buffer...")
    parser.parse_buffer(uff_model, network)
    print("starting building an engine...")
    engine = builder.build_cuda_engine(network)
    print("finished building an engine...")
    if engine is not None:
        with open(tensorrt_model_path, "wb") as f:
            f.write(engine.serialize())
    else:
        print("no engine built :(")
