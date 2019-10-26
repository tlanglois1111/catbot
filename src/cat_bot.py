import traitlets
import time
from IPython.display import display
from jetbot import Camera, bgr8_to_jpeg

# init camera
print('initialize camera')
camera = Camera.instance(width=1280, height=720)
cat_count = 0

# create save directory
import os
image_dir = 'dataset/cats'

# we have this "try/except" statement because these next functions can throw an error if the directories exist already
try:
    os.makedirs(image_dir)
except FileExistsError:
    print('Directories not created because they already exist')

# save image methods
from uuid import uuid1

def save_snapshot(directory, image):
    print('saving snapshot #', cat_count)
    image_path = os.path.join(directory, str(uuid1()) + '.jpg')
    with open(image_path, 'wb') as f:
        f.write(image)

def save_image(image):
    global image_dir, cat_count
    save_snapshot(image_dir, image)
    cat_count = len(os.listdir(image_dir))
    print(cat_count)

from jetbot import ObjectDetector

print('loading mobilenet_v2')
model = ObjectDetector('ssd_mobilenet_v2_coco.engine')
print('done')

# setup models
import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np

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
from jetbot import Robot
robot = Robot()

# run bot
width = 300
height = 300

def detection_center(detection):
    #Computes the center x, y coordinates of the object
    bbox = detection['bbox']
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    return (center_x, center_y)

def norm(vec):
    #Computes the length of the 2D vector
    return np.sqrt(vec[0]**2 + vec[1]**2)

def closest_detection(detections):
    #Finds the detection closest to the image center
    closest_detection = None
    for det in detections:
        center = detection_center(det)
        if closest_detection is None:
            closest_detection = det
        elif norm(detection_center(det)) < norm(detection_center(closest_detection)):
            closest_detection = det
    return closest_detection

def execute(change):
    cur_time = time.time()
    last_save = 0

    image = change['new']

    # execute collision model to determine if blocked
    collision_output = collision_model(preprocess(image)).detach().cpu()
    prob_blocked = float(F.softmax(collision_output.flatten(), dim=0)[0])

    # turn left if blocked
    #if prob_blocked > 0.5:
        #robot.left(0.3)
        #return

    # compute all detected objects
    detections = model(image)

    # draw all detections on image
    print(detections[0])
    for det in detections[0]:
        bbox = det['bbox']
        #cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)

    # select detections that match selected class label
    matching_detections = [d for d in detections[0] if d['label'] == 17]

    # get detection closest to center of field of view and draw it
    det = closest_detection(matching_detections)
    if det is not None:
        print('detected cat')
        bbox = det['bbox']
        #cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), (int(width * bbox[2]), int(height * bbox[3])), (0, 255, 0), 5)
        if (cur_time - last_save) > 5000:
            last_save = cur_time
            save_image(bgr8_to_jpeg(image))
        else:
            print('too soon to save another image')

        print('center detection')
        center = detection_center(det)
        robot.set_motors(
            float(0 + 0.4 * center[0]),
            float(0 - 0.4 * center[0])
        )

print('start cat hunt')
execute({'new': camera.value})

camera.unobserve_all()
camera.observe(execute, names='value')
