"""camera_tf_trt.py

This is a Camera TensorFlow/TensorRT Object Detection code for
Jetson Nano.
"""

import sys
import time
import ctypes
import argparse
import os
import threading
import math

import logging
import logging.config

import numpy as np
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import pycuda.driver as cuda
import tensorrt as trt
import RTIMU

from utils.ssd_classes import get_cls_dict
from utils.camera import add_camera_args, Camera

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
            'maxBytes': 100000,
            'backupCount': 3
        },
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'WARN',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers': {
        '__main__': {
            'handlers': ['default_handler'],
            'level': 'INFO',
            'propagate': False
        }
    },
    'root': {
        'handlers': ['default_handler'],
        'level': 'WARN',
        'propagate': False
    }
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)
# Ask tensorflow logger not to propagate logs to parent (which causes
# duplicated logging)
logging.getLogger('tensorflow').propagate = False

# Constants
v2_coco_labels_to_capture = [16, 17, 18]
INPUT_WH = (300, 300)
OUTPUT_LAYOUT = 7
SUPPORTED_MODELS = [
    'ssd_mobilenet_v1_coco',
    'ssd_mobilenet_v1_egohands',
    'ssd_mobilenet_v2_coco',
    'ssd_mobilenet_v2_egohands',
]


def parse_args():
    """Parse input arguments."""
    desc = 'Follow cats with SSD model on Jetson Nano'
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--model', type=str, default='ssd_mobilenet_v1_coco', choices=SUPPORTED_MODELS)
    args = parser.parse_args()
    return args


def preprocess(img):
    """Preprocess an image before SSD inferencing."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_WH)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img = (2.0/255.0) * img - 1.0
    return img


def postprocess(img, output, conf_th):
    """Postprocess TRT SSD output."""
    img_h, img_w, _ = img.shape
    boxes, confs, clss = [], [], []
    for prefix in range(0, len(output), OUTPUT_LAYOUT):
        #index = int(output[prefix+0])
        conf = float(output[prefix+2])
        if conf < conf_th:
            continue
        x1 = int(output[prefix+3] * img_w)
        y1 = int(output[prefix+4] * img_h)
        x2 = int(output[prefix+5] * img_w)
        y2 = int(output[prefix+6] * img_h)
        cls = int(output[prefix+1])
        boxes.append((x1, y1, x2, y2))
        confs.append(conf)
        clss.append(cls)
    return boxes, confs, clss


class TrtSSD(object):
    """TrtSSD class encapsulates things needed to run TRT SSD."""

    def _load_plugins(self):
        ctypes.CDLL("../dataset/ssd/libflattenconcat.so")
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self):
        TRTbin = '../dataset/ssd/TRT_%s.bin' % self.model
        logger.info("loading model %s" % TRTbin)
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self):
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return self.engine.create_execution_context()

    def __init__(self, model):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()
        self.engine = self._load_engine()

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.context = self._create_context()

    def __del__(self):
        """Free CUDA memories."""
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs

    def detect(self, img, conf_th=0.3):
        """Detect objects in the input image."""
        img_resized = preprocess(img)
        np.copyto(self.host_inputs[0], img_resized.ravel())

        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(
            self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        output = self.host_outputs[0]
        return postprocess(img, output, conf_th)


def detection_center(detection, width, height):
    # Computes the center x, y coordinates of the object
    bbox = detection['bbox']
    center_x = (bbox[0] + bbox[2]) / width / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / height / 2.0 - 0.5
    return (center_x, center_y)


def norm(vec):
    # Computes the length of the 2D vector
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2)


def closest_detection(detections, width, height):
    # Finds the detection closest to the image center
    closest_detection = None
    for det in detections:
        center = detection_center(det, width, height)
        if closest_detection is None:
            closest_detection = det
        elif norm(detection_center(det, width, height)) < norm(detection_center(closest_detection, width, height)):
            closest_detection = det
    return closest_detection


def loop_and_detect(cam, trt_ssd, conf_th, robot, model):
    """Loop, grab images from camera, and do object detection.

    # Arguments
      cam: the camera object (video source).
      tf_sess: TensorFlow/TensorRT session to run SSD object detection.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    settings_path = "../dataset/RTIMULib"
    logger.info("Using settings file %s", settings_path + ".ini")
    if not os.path.exists(settings_path + ".ini"):
        logger.error("Settings file does not exist, will be created")

    s = RTIMU.Settings(settings_path)
    imu = RTIMU.RTIMU(s)

    logger.info("IMU Name: %s", imu.IMUName())

    if not imu.IMUInit():
        logger.error("IMU Init Failed")
    else:
        logger.info("IMU Init Succeeded")
        imu.setSlerpPower(0.02)
        imu.setGyroEnable(True)
        imu.setAccelEnable(True)
        imu.setCompassEnable(True)

        poll_interval = imu.IMUGetPollInterval()
        logger.info("Recommended Poll Interval: %f", poll_interval)

    cls_dict = get_cls_dict(model.split('_')[-1])
    fps = 0.0
    counter = 58
    tic = time.time()
    while True:
        gyro = imu.getIMUData().copy()
        logger.info(gyro)
        img = cam.read()
        if img is not None:
            boxes, confs, clss = trt_ssd.detect(img, conf_th)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.9 + curr_fps*0.1)
            tic = toc

            counter += 1
            if counter > fps:
                logger.info("fps: %f", fps)
                if imu.IMURead():
                    accel = gyro["accel"]
                    logger.info(accel)
                counter = 0

            # compute all detected objects
            detections = []
            for i, (bb, cf, cl) in enumerate(zip(boxes, confs, clss)):
                detections.append({'bbox': bb, 'confidence': cf, 'label': int(cl)})
            if logger.isEnabledFor(logging.DEBUG) and (len(detections)) > 0:
                logger.debug(detections)

            # select detections that match selected class label
            matching_detections = [d for d in detections if d['label'] in v2_coco_labels_to_capture and d['confidence'] > 0.50]

            if len(matching_detections) > 0:
                logger.debug(matching_detections)

            # get detection closest to center of field of view and center bot
            det = closest_detection(matching_detections, width=cam.img_width, height=cam.img_height)
            if det is not None:
                center = detection_center(det, cam.img_width, cam.img_height)
                logger.info("center: %s, on object: %s", center, cls_dict[det['label']])

                move_speed = 2.0 * center[0]
                if abs(move_speed) > 0.3:
                    if move_speed > 0.0:
                        robot.right(move_speed)
                    else:
                        robot.left(abs(move_speed))
            else:
                robot.stop()


def main():
    args = parse_args()
    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    trt_ssd = TrtSSD(args.model)

    cam.start()

    # initialize bot
    logger.info('initialize robot')
    robot = Robot()

    logger.info('starting to loop and detect')
    loop_and_detect(cam=cam, trt_ssd=trt_ssd, conf_th=0.3, robot=robot, model=args.model)

    logger.info('cleaning up')
    robot.stop()
    cam.stop()
    cam.release()


if __name__ == '__main__':
    main()
