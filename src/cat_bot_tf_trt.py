"""camera_tf_trt.py

This is a Camera TensorFlow/TensorRT Object Detection code for
Jetson Nano.
"""


import sys
import time
import logging
import logging.config
import argparse

import numpy as np
import tensorflow as tf

from utils.camera import add_camera_args, Camera
from utils.od_utils import read_label_map, build_trt_pb, load_trt_pb, detect
from utils.visualization import BBoxVisualization
from jetbot import Robot

logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        }
    },
    'handlers': {
        'default_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'filename': '../logs/cat_bot.log',
            'maxBytes': 10000,
            'backupCount': 3
        },
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers': {
        '__main__': {
            'handlers': ['default_handler'],
            'level': 'DEBUG',
            'propagate': False
        }
    },
    'root': {
        'handlers': ['default_handler'],
        'level': 'INFO',
        'propagate': False
    }
}

# Constants
DEFAULT_MODEL = '../dataset/tf/frozen_inference_graph'
DEFAULT_CONFIG = '../dataset/tf/ssd_mobilenet_v2_coco.config'
DEFAULT_LABELMAP = '../dataset/tf/label_map.pbtxt'
DEFAULT_CHECKPOINT = '../dataset/tf/model-ckpt-28553'
v2_coco_labels_to_capture = [1, 2]

def parse_args():
    """Parse input arguments."""
    desc = ('This script captures and displays live camera video, '
            'and does real-time object detection with TF-TRT model '
            'on Jetson TX2/TX1/Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--model', dest='model',
                        help='tf-trt object detection model '
                             '[{}]'.format(DEFAULT_MODEL),
                        default=DEFAULT_MODEL, type=str)
    parser.add_argument('--config', dest='config',
                        help='model config '
                             '[{}]'.format(DEFAULT_MODEL),
                        default=DEFAULT_CONFIG, type=str)
    parser.add_argument('--build', dest='do_build',
                        help='re-build TRT pb file (instead of using'
                        'the previously built version)',
                        default=False, type=bool)
    parser.add_argument('--labelmap', dest='labelmap_file',
                        help='[{}]'.format(DEFAULT_LABELMAP),
                        default=DEFAULT_LABELMAP, type=str)
    parser.add_argument('--confidence', dest='conf_th',
                        help='confidence threshold [0.3]',
                        default=0.3, type=float)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint path' 
                             '[{}]'.format(DEFAULT_CHECKPOINT),
                        default=DEFAULT_CHECKPOINT, type=str)
    args = parser.parse_args()
    return args


def detection_center(detection):
    # Computes the center x, y coordinates of the object
    bbox = detection['bbox']
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    return (center_x, center_y)


def norm(vec):
    # Computes the length of the 2D vector
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2)


def closest_detection(detections):
    # Finds the detection closest to the image center
    closest_detection = None
    for det in detections:
        center = detection_center(det)
        if closest_detection is None:
            closest_detection = det
        elif norm(detection_center(det)) < norm(detection_center(closest_detection)):
            closest_detection = det
    return closest_detection


def loop_and_detect(cam, tf_sess, conf_th, od_type, robot, logger):
    """Loop, grab images from camera, and do object detection.

    # Arguments
      cam: the camera object (video source).
      tf_sess: TensorFlow/TensorRT session to run SSD object detection.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    fps = 0.0
    counter=58
    tic = time.time()
    while True:
        img = cam.read()
        if img is not None:
            box, conf, cls = detect(img, tf_sess, conf_th, od_type=od_type)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.9 + curr_fps*0.1)
            tic = toc

            counter += 1
            if counter > 60:
                logger.log("fps: %f", fps)
                counter = 0

            # compute all detected objects
            detections = []
            i = 0
            for bb, cf, cl in zip(box, conf, cls):
                detections[i] = {'bbox': bb, 'confidence': cf, 'label': int(cl)}
            if logger.isEnabledFor(logging.DEBUG) and (len(detections)) > 0:
                logger.debug(detections)

            # select detections that match selected class label
            matching_detections = [d for d in detections if d['label'] in v2_coco_labels_to_capture and d['confidence'] > 0.50]

            tf_image_list = []
            if len(matching_detections) > 0:
                logger.info(matching_detections)

            # get detection closest to center of field of view and center bot
            det = closest_detection(matching_detections)
            if det is not None:
                center = detection_center(det)
                logger.debug("center: %s", center)

                move_speed = 2.0 * center[0]
                if abs(move_speed) > 0.3:
                    if move_speed > 0.0:
                        robot.right(move_speed)
                    else:
                        robot.left(abs(move_speed))
            else:
                robot.stop()


def main():
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)
    # Ask tensorflow logger not to propagate logs to parent (which causes
    # duplicated logging)
    logging.getLogger('tensorflow').propagate = False

    args = parse_args()
    logger.info('called with args: %s' % args)

    # build the class (index/name) dictionary from labelmap file
    #logger.info('reading label map')
    #cls_dict = read_label_map(args.labelmap_file)

    pb_path = '{}.pb'.format(args.model)
    if args.do_build:
        logger.info('building TRT graph and saving to pb: %s' % pb_path)
        build_trt_pb(args.config, pb_path, args.checkpoint)

    logger.info('opening camera device/file')
    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    logger.info('loading TRT graph from pb: %s' % pb_path)
    trt_graph = load_trt_pb(pb_path)

    logger.info('starting up TensorFlow session')
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config, graph=trt_graph)

    logger.info('warming up the TRT graph with a dummy image')
    od_type = 'faster_rcnn' if 'faster_rcnn' in args.model else 'ssd'
    dummy_img = np.zeros((600, 600, 3), dtype=np.uint8)
    _, _, _ = detect(dummy_img, tf_sess, conf_th=.3, od_type=od_type)

    cam.start()  # ask the camera to start grabbing images

    # initialize bot
    logger.info('initialize robot')
    robot = Robot()

    # grab image and do object detection (until stopped by user)
    logger.info('starting to loop and detect')
    loop_and_detect(cam, tf_sess, args.conf_th, od_type=od_type, robot=robot, logger=logger)

    logger.info('cleaning up')
    robot.stop()
    cam.stop()  # terminate the sub-thread in camera
    tf_sess.close()
    cam.release()


if __name__ == '__main__':
    main()
