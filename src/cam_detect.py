import numpy as np
import os
import tensorflow as tf
import cv2
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

# Define the video stream
cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN = '../dataset/tf/catbot-detection-graphs/catbot_detection_graph_v1.pb/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('../dataset/tf', 'label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 2
input_names = ['image_tensor']


def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


# The TensorRT inference graph file downloaded from Colab or your local machine.
detection_graph = get_frozen_graph(PATH_TO_FROZEN)

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


input_names = ['image_tensor']

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

# Detection
with tf.Session(config=tf_config) as sess:
    tf.import_graph_def(detection_graph, name='')

    while True:

        # Read frame from camera
        ret, image_np = cap.read()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Extract image tensor
        image_tensor = sess.graph.get_tensor_by_name(input_names[0]+':0')
        # Extract detection boxes
        boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
        # Extract detection scores
        scores = sess.graph.get_tensor_by_name('detection_scores:0')
        # Extract detection classes
        classes = sess.graph.get_tensor_by_name('detection_classes:0')
        # Extract number of detectionsd
        num_detections = sess.graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                           np.squeeze(boxes),
                                                           np.squeeze(classes).astype(np.int32),
                                                           np.squeeze(scores),
                                                           category_index,
                                                           use_normalized_coordinates=True,
                                                           line_thickness=8)
        # Display output
        cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
