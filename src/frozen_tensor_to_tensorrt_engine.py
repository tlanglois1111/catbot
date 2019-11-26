from tf_trt_models.detection import build_detection_graph
from tensorflow.python.compiler.tensorrt import trt_convert as convert
import tensorrt as trt
import tensorflow as tf
import numpy as np
import cv2
import uff

def save_image(data, fname="../dataset/img.png", swap_channel=True):
    if swap_channel:
        data = data[..., ::-1]
    cv2.imwrite(fname, data)


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]),
                  (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)


def non_max_suppression(boxes, probs=None, nms_threshold=0.3):
    """Non-max suppression

    Arguments:
        boxes {np.array} -- a Numpy list of boxes, each one are [x1, y1, x2, y2]
    Keyword arguments
        probs {np.array} -- Probabilities associated with each box. (default: {None})
        nms_threshold {float} -- Overlapping threshold 0~1. (default: {0.3})

    Returns:
        list -- A list of selected box indexes.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap < nms_threshold)[0])))
    # return only the bounding boxes indexes
    return pick


config_path = "../dataset/tf/ssd_mobilenet_v2_coco.config"
#frozen_path = "../dataset/tf/trained-inference-graphs/frozen_inference_graph.pb"
frozen_path = "../dataset/tf/catbot-detection-graphs/frozen_inference_graph.pb"
checkpoint_path = "../dataset/tf/model.ckpt-11777"
uff_model_path = "../dataset/tf/catbot-detection-graphs/catbot.uff"
tensorrt_model_path = "../dataset/tf/catbot-detection-graphs/catbot.engine"
output_names = ['detection_boxes', 'detection_classes', 'detection_scores', 'num_detections']
input_names = ['image_tensor']

use_frozen_graph = False

if use_frozen_graph:
    with tf.Session() as sess:
        # First deserialize your frozen graph:
        with tf.gfile.GFile(frozen_path,"rb") as f:
            frozen_graph = tf.GraphDef()
            frozen_graph.ParseFromString(f.read())
else:
    frozen_graph, input_names, output_names = build_detection_graph(
        config=config_path,
        checkpoint=checkpoint_path,
        score_threshold=0.3,
        batch_size=1)

print("input names: {s}",input_names)
print("output names: {s}",output_names)

uff_model = uff.from_tensorflow(graphdef=frozen_graph,
                                output_filename=uff_model_path,
                                #preprocessor="remove_asserts.py",
                                text=True,
                                return_graph_info=False)

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
    #parser.register_input(input_names[0], (3, 100, 100), trt.UffInputOrder.NHWC)
    for output_node in output_names:
        parser.register_output(output_node)
    print("parsing buffer...")
    if parser.parse_buffer(uff_model, network):
        print("starting building an engine...")
        engine = builder.build_cuda_engine(network)
        print("finished building an engine...")
        if engine is not None:
            with open(tensorrt_model_path, "wb") as f:
                f.write(engine.serialize())
    else:
        print("no engine built :(")





# test it
trt_graph = convert.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

if not use_frozen_graph:
    with open(frozen_path, "wb") as f:
        f.write(trt_graph.SerializeToString())


# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

#IMAGE_PATH = "../dataset/cats/77f44f40-fb63-11e9-9404-72b5f773b75d.jpg"
IMAGE_PATH = "../dataset/cats/fbbcb7b6-fcc8-11e9-aacc-72b5f773b75d.jpg"
image = cv2.imread(IMAGE_PATH)
image = cv2.resize(image, (300, 300))

scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
    tf_input: image[None, ...]
})
boxes = boxes[0]  # index by 0 to remove batch dimension
scores = scores[0]
classes = classes[0]
num_detections = int(num_detections[0])

# Boxes unit in pixels (image coordinates).
boxes_pixels = []
for i in range(num_detections):
    # scale box to image coordinates
    box = boxes[i] * np.array([image.shape[0],
                               image.shape[1], image.shape[0], image.shape[1]])
    box = np.round(box).astype(int)
    boxes_pixels.append(box)
boxes_pixels = np.array(boxes_pixels)

# Remove overlapping boxes with non-max suppression, return picked indexes.
pick = non_max_suppression(boxes_pixels, scores[:num_detections], 0.5)


for i in pick:
    box = boxes_pixels[i]
    box = np.round(box).astype(int)
    # Draw bounding box.
    image = cv2.rectangle(
        image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
    label = "{}:{:.2f}".format(int(classes[i]), scores[i])
    # Draw label (class index and probability).
    draw_label(image, (box[1], box[0]), label)

# Display the labeled image.
save_image(image[:, :, ::-1])




