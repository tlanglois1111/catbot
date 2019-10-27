import time
import signal
import sys
import threading
import os
import torch
import torchvision
import cv2
import numpy as np
from uuid import uuid1
from jetbot import Camera, bgr8_to_jpeg, ObjectDetector, Robot


# init camera
print('initialize camera')

camera_width = 300
camera_height = 300
model_width = 300
model_height = 300
xscale = model_width * (camera_width / model_width)
yscale = model_height * (camera_height / model_height)
camera = Camera.instance(width=camera_width, height=camera_height, capture_width=3280, capture_height=2464)  # W = 3280 H = 2464   1920 x 1080
cat_count = 0
debug = False

# create save directory

image_dir = 'dataset/cats'

# we have this "try/except" statement because these next functions can throw an error if the directories exist already
try:
    os.makedirs(image_dir)
except FileExistsError:
    print('Directories not created because they already exist')


# Inherting the base class 'Thread'
class AsyncWrite(threading.Thread):

    def __init__(self, directory, image):
        # calling superclass init
        threading.Thread.__init__(self)
        self.image = image
        self.directory = directory

    def run(self):
        global cat_count

        image_path = os.path.join(self.directory, str(uuid1()) + '.jpg')
        with open(image_path, 'wb') as f:
            f.write(self.image)
        cat_count = len(os.listdir(image_dir))
        print('saved snapshot: ', cat_count)


def save_image(image):
    global image_dir, cat_count
    background = AsyncWrite(image_dir, image)
    background.start()


print('loading mobilenet_v2')
model = ObjectDetector('ssd_mobilenet_v2_coco.engine')
print('done')

# setup models
collision_model = torchvision.models.alexnet(pretrained=False)
collision_model.classifier[6] = torch.nn.Linear(collision_model.classifier[6].in_features, 2)
print('load collision model')
collision_model.load_state_dict(torch.load('best_model.pth'))
print('done loading')
device = torch.device('cuda')
collision_model = collision_model.to(device)

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)


def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.resize(x, (224, 224))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x


# initialize bot
print('initialize robot')

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


def execute(change):
    global last_save, xscale, yscale

    cur_time = time.time()

    image = change['new']
    resized_image = image
    if model_width != camera_width and model_height != camera_height:
        resized_image = cv2.resize(image, (model_width, model_height))

    # execute collision model to determine if blocked
    #collision_output = collision_model(preprocess(image)).detach().cpu()
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
    matching_detections = [d for d in detections[0] if d['label'] in [1, 16, 17, 18] and d['confidence'] > 0.50]

    # get detection closest to center of field of view and draw it
    det = closest_detection(matching_detections)
    if det is not None:
        print('detected cat')
        print(detections[0])
        bbox = det['bbox']
        # cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])),(int(width * bbox[2]), int(height * bbox[3])), (0, 255, 0), 5)
        if (cur_time - last_save) > 5.0:
            last_save = cur_time

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
            crop = image[ymin:ymax, xmin:xmax]

            save_image(bgr8_to_jpeg(crop))
        else:
            print('too soon to save another image')

        print('center detection')
        center = detection_center(det)
        print("center:", center)
        if center[0] > 0.0:
            robot.right(2.0 * center[0])
        else:
            robot.left(abs(2.0 * center[0]))
        #robot.set_motors(
        #    float(0.4 + 0.4 * center[0]),
        #    float(0.4 - 0.4 * center[1])
        #)
        #time.sleep(0.2)

    else:
        robot.stop()
        if debug and (len(detections[0])) > 0:
            print(detections[0])


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    shutdown()


signal.signal(signal.SIGINT, signal_handler)

print('start cat hunt')
#while True:
#    execute({'new': camera.value})

robot.stop()
camera.unobserve_all()
print('calling observe')
camera.observe(execute, names='value')
