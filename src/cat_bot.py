import time
import signal
import sys
import threading
import os
import logging
import logging.config
import csv
import torch
import torchvision
import cv2
import numpy as np
from uuid import uuid1
from jetbot import Camera, bgr8_to_jpeg, ObjectDetector, Robot

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
            'filename': 'cat_bot.log',
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
            'level': 'INFO',
            'propagate': False
        }
    },
    'root': {
        'handlers': ['default_handler'],
        'level': 'INFO',
        'propagate': False
    }
}


logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

# init camera
logger.info('initialize camera')

camera_width = 600
camera_height = 600
model_width = 300
model_height = 300
xscale = model_width * (camera_width / model_width)
yscale = model_height * (camera_height / model_height)
#camera = Camera.instance(width=camera_width, height=camera_height, capture_width=3280, capture_height=2464)  # W = 3280 H = 2464   1920 x 1080   1280 x 720
camera = Camera.instance(width=camera_width, height=camera_height, capture_width=1920, capture_height=1080)  # W = 3280 H = 2464   1920 x 1080   1280 x 720
cat_count = 0
seconds_between_pics = 1.0
v2_coco_labels_to_capture = [16, 17, 18]


# create save directory

image_dir = 'dataset/cats'

# we have this "try/except" statement because these next functions can throw an error if the directories exist already
try:
    os.makedirs(image_dir)
except FileExistsError:
    logger.info('image directories not created because they already exist')


# Inherting the base class 'Thread'
class AsyncWrite(threading.Thread):

    def __init__(self, directory, filename, image, tf_list):
        # calling superclass init
        threading.Thread.__init__(self)
        self.image = image
        self.directory = directory
        self.filename = filename
        self.tf_list = tf_list

    def run(self):
        global cat_count, debug

        image_path = os.path.join(self.directory, self.filename+'.jpg')
        with open(image_path, 'wb') as f:
            f.write(self.image)
        cat_count = len(os.listdir(image_dir))
        logger.debug('saved snapshot: %d', cat_count)

        csv_path = os.path.join(self.directory, self.filename+'.csv')
        with open(csv_path, 'a') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerows(self.tf_list)
            outcsv.close()


def save_image(image, filename, tf_list):
    global image_dir, cat_count
    background = AsyncWrite(image_dir, filename, image, tf_list)
    background.start()


logger.info('loading ssd_mobilenet_v2_coco')
model = ObjectDetector('ssd_mobilenet_v2_coco.engine')

# setup models
collision_model = torchvision.models.alexnet(pretrained=False)
collision_model.classifier[6] = torch.nn.Linear(collision_model.classifier[6].in_features, 2)
logger.info('load collision model')
collision_model.load_state_dict(torch.load('best_model.pth'))
logger.info('done loading')
device = torch.device('cuda')
collision_model = collision_model.to(device)

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)


def preprocess(camera_value, width, height):
    global device, normalize
    x = camera_value
    x = cv2.resize(x, (width, height))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x


# initialize bot
logger.info('initialize robot')

robot = Robot()


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


last_save = 0.0


def shutdown():
    camera.unobserve_all()
    time.sleep(1.0)
    robot.stop()
    sys.exit(0)

ping_counter = 58
def execute(change):
    global last_save, xscale, yscale, seconds_between_pics, debug, v2_coco_labels_to_capture, model_width, model_height, ping_counter

    ping_counter += 1
    if ping_counter > 60:
        logger.info("still observing")
        ping_counter = 0

    image = change['new']
    resized_image = image
    if model_width != camera_width and model_height != camera_height:
        resized_image = cv2.resize(image, (model_width, model_height), interpolation=cv2.INTER_AREA)

    # execute collision model to determine if blocked
    #collision_output = collision_model(preprocess(image,model_224,224)).detach().cpu()
    #prob_blocked = float(F.softmax(collision_output.flatten(), dim=0)[0])
    # turn left if blocked
    #if prob_blocked > 0.5:
    #    robot.left(0.4)
    #    return

    # compute all detected objects
    detections = model(resized_image)

    # draw all detections on image
    # for det in detections[0]:
    #     bbox = det['bbox']
    #     cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)

    # select detections that match selected class label
    matching_detections = [d for d in detections[0] if d['label'] in v2_coco_labels_to_capture and d['confidence'] > 0.50]

    tf_image_list = []
    if len(matching_detections) > 0:
        logger.info(matching_detections)

        filename = str(uuid1())
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
            if xmax > camera_width:
                xmax = camera_width-1
            if ymax > camera_height:
                ymax = camera_height-1

            tf_image_desc = [filename+'.jpg', camera_width, camera_height, int(d['label']), xmin, ymin, xmax, ymax]
            tf_image_list.append(tf_image_desc)

        save_image(bgr8_to_jpeg(image), filename, tf_image_list)

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
        if logger.isEnabledFor(logging.DEBUG) and (len(detections[0])) > 0:
            logger.debug(detections[0])


def signal_handler(sig, frame):
    logger.info('You pressed Ctrl+C!')
    shutdown()


signal.signal(signal.SIGINT, signal_handler)

logger.info('start cat hunt')
robot.stop()
camera.unobserve_all()
logger.info('calling observe')
camera.observe(execute, names='value')

#while True:
#    execute({'new': camera.value})

