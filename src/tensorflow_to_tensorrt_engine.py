from tf_trt_models.detection import build_detection_graph
import uff
import tensorrt as trt
import graphsurgeon as gs
import tensorflow as tf

config_path = "../dataset/tf/ssd_mobilenet_v2_coco.config"
frozen_path = "../dataset/tf/catbot-detection-graphs/catbot_detection_graph_v1.pb/frozen_inference_graph.pb"
checkpoint_path = "../dataset/tf/model.ckpt-24082"
uff_model_path = "../dataset/tf/catbot-detection-graphs/catbot_detection_graph_v1.pb/catbot.uff"
tensorrt_model_path = "../dataset/tf/catbot-detection-graphs/catbot_detection_graph_v1.pb/catbot.engine"
output_names = ['detection_boxes', 'detection_classes', 'detection_scores', 'num_detections']
input_names = ['image_tensor']


def add_plugin(graph):
    all_assert_nodes = graph.find_nodes_by_op("Assert")
    graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)

    return graph


def prepare_namespace_plugin_map():
    # In this sample, the only operation that is not supported by TensorRT
    # is tf.nn.relu6, so we create a new node which will tell UffParser which
    # plugin to run and with which arguments in place of tf.nn.relu6.


    # The "clipMin" and "clipMax" fields of this TensorFlow node will be parsed by createPlugin,
    # and used to create a CustomClipPlugin with the appropriate parameters.
    trt_equal = gs.create_plugin_node(name="trt_equal", op="EqualPlugin", in_width=300.0, in_height=300.0, in_channel=3.0)
    trt_select = gs.create_plugin_node(name="trt_select", op="SelectPlugin", in_width=300.0, in_height=300.0, value=0.0)
    namespace_plugin_map = {
        "Select": trt_select,
        "Equal": trt_equal,
        "_Equal": trt_equal
    }
    return namespace_plugin_map


#if not tf.gfile.Exists(frozen_path):
frozen_graph, input_names, output_names = build_detection_graph(
    config=config_path,
    checkpoint=checkpoint_path,
    score_threshold=0.3,
    batch_size=1)

print("input names: {s}",input_names)
print("output names: {s}",output_names)

with tf.gfile.GFile(frozen_path, "wb") as f:
    f.write(frozen_graph.SerializeToString())

#dynamic_graph = add_plugin(gs.DynamicGraph(frozen_path))
#uff_model = uff.from_tensorflow(graphdef=frozen_graph, output_nodes=output_names, output_filename=uff_model_path, list_nodes=True, text=True)

dynamic_graph = add_plugin(gs.DynamicGraph(frozen_graph))
dynamic_graph.collapse_namespaces(prepare_namespace_plugin_map())
uff_model = uff.from_tensorflow(graphdef=frozen_graph,
                                output_names=output_names,
                                output_filename=uff_model_path,
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
