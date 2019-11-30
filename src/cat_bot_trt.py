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

import csv
from uuid import uuid1

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

from jetbot import Robot, bgr8_to_jpeg

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
            'level': 'DEBUG',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers': {
        '__main__': {
            'handlers': ['console'],
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
model_width = 300
model_height = 300
OUTPUT_LAYOUT = 7
SUPPORTED_MODELS = [
    'ssd_mobilenet_v1_coco',
    'ssd_mobilenet_v1_egohands',
    'ssd_mobilenet_v2_coco',
    'ssd_mobilenet_v2_egohands',
]
IMAGE_DIR = '../dataset/cats'
FORWARD_SPEED = 0.7
BACKWARD_SPEED = -0.6
TURNING_SPEED = 0.6
REVERSE_TIME = 0.7
BLOCKED_THRESHOLD = 0.01

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
    img = cv2.resize(img, (model_width, model_height))
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


# Inherting the base class 'Thread'
class AsyncWrite(threading.Thread):

    def __init__(self, directory, filename, image, tf_list, blocked):
        # calling superclass init
        threading.Thread.__init__(self)
        self.image = image
        self.directory = directory
        self.filename = filename
        self.tf_list = tf_list
        self.blocked = blocked

    def run(self):
        if len(self.tf_list) == 0:
            if self.blocked:
                path = self.directory + "/blocked"
            else:
                path = self.directory + "/not_blocked"
            image_path = os.path.join(path, self.filename+'.jpg')
            with open(image_path, 'wb') as f:
                f.write(self.image)
        else:
            image_path = os.path.join(self.directory, self.filename+'.jpg')
            with open(image_path, 'wb') as f:
                f.write(self.image)

            csv_path = os.path.join(self.directory, self.filename+'.csv')
            with open(csv_path, 'a') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerows(self.tf_list)
                outcsv.close()


def save_image(image, filename, tf_list=[], blocked=False):
    background = AsyncWrite(IMAGE_DIR, filename, image, tf_list, blocked)
    background.start()


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


moving = False
acceleration = [[0, 0, 0], [0, 0, 0]]
velocity = [0, 0, 0]


def get_velocity(gyro):
    for j in range(0, 3):
        acceleration[0] = acceleration[1]
        acceleration[1] = gyro["accel"]

        velocity[j] = acceleration[0][j] + ((acceleration[1][j] - acceleration[0][j]) / 2)
        #position[j][1] = position[j][0] + velocity[j][0] + ((velocity[j][1] - velocity[j][0]) / 2)

        return velocity


def loop_and_detect(cam, trt_ssd, conf_th, robot, model):
    global moving
    """Loop, grab images from camera, and do object detection.

    # Arguments
      cam: the camera object (video source).
      tf_sess: TensorFlow/TensorRT session to run SSD object detection.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """

    xscale = model_width * (cam.img_width / model_width)
    yscale = model_height * (cam.img_height / model_height)

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
        img = cam.read()
        filename = str(uuid1())
        if imu.IMURead():
            gyro = imu.getIMUData().copy()

            v = get_velocity(gyro)

            accel = gyro["accel"]
            fusion = gyro["fusionPose"]
            compass = gyro["compass"]
            gyro1 = gyro["gyro"]
            fusionq = gyro["fusionQPose"]
            logger.info("velocity:  x: %.4f y: %.4f z: %.4f" % (v[0], v[1], v[2]))
            logger.info("  fusion:  x: %.4f y: %.4f z: %.4f" % (fusion[0], fusion[1], fusion[2]))
            logger.info("   accel:  x: %.4f y: %.4f z: %.4f" % (accel[0], accel[1], accel[2]))
            logger.info(" compass:  x: %.4f y: %.4f z: %.4f" % (compass[0], compass[1], compass[2]))
            logger.info("    gyro:  x: %.4f y: %.4f z: %.4f" % (gyro1[0], gyro1[1], gyro1[2]))
            logger.info(" fusionq:  x: %.4f y: %.4f z: %.4f" % (fusionq[0], fusionq[1], fusionq[2]))
        else:
            gyro = []

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
                if len(gyro) > 0:
                    if img is not None and moving and accel[0] > BLOCKED_THRESHOLD:
                        save_image(bgr8_to_jpeg(img), filename, blocked=False)
                        logger.info("not blocked:  x: %.4f y: %.4f z: %.4f" % (accel[0], accel[1], accel[2]))
                        robot.forward(FORWARD_SPEED)
                        moving = True
                    elif img is not None and moving:
                        if False:
                            save_image(bgr8_to_jpeg(img), filename, blocked=True)
                            logger.info("blocked:  x: %.4f y: %.4f z: %.4f" % (accel[0], accel[1], accel[2]))
                            moving = False
                            robot.set_motors(BACKWARD_SPEED, BACKWARD_SPEED/2)
                            time.sleep(REVERSE_TIME)
                counter = 0

            # compute all detected objects
            detections = []
            for i, (bb, cf, cl) in enumerate(zip(boxes, confs, clss)):
                detections.append({'bbox': bb, 'confidence': cf, 'label': int(cl)})
            if logger.isEnabledFor(logging.DEBUG) and (len(detections)) > 0:
                logger.debug(detections)

            # select detections that match selected class label
            matching_detections = [d for d in detections if d['label'] in v2_coco_labels_to_capture and d['confidence'] > 0.50]

            # save detected image for later training
            if len(matching_detections) > 0:
                logger.debug(matching_detections)

            tf_image_list = []
            for d in matching_detections:
                bbox = d['bbox']
                xmin = int(xscale * bbox[0])
                ymin = int(yscale * bbox[1])
                xmax = int(xscale * bbox[2])
                ymax = int(yscale * bbox[3])
                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax > cam.img_width:
                    xmax = cam.img_width-1
                if ymax > cam.img_height:
                    ymax = cam.img_height-1

                tf_image_desc = [filename+'.jpg', cam.img_width, cam.img_height, int(d['label']), xmin, ymin, xmax, ymax]
                tf_image_list.append(tf_image_desc)

            save_image(bgr8_to_jpeg(img), filename, tf_list=tf_image_list)
            # end save image

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

        robot.forward(FORWARD_SPEED)
        moving = True


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
