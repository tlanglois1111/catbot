"""camera_tf_trt.py

This is a Camera TensorFlow/TensorRT Object Detection code for
Jetson Nano.
"""

import sys
import time
import ctypes
import argparse
import os
import math

import logging
import logging.config

import RTIMU

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

class Gyro(object):
    """RTIMU encapsulates use of gyro device"""

    def load_rtimu_lib(self):

        logger.info("Using settings file %s", self.SETTINGS_FILE + ".ini")
        if not os.path.exists(self.SETTINGS_FILE + ".ini"):
            logger.error("Settings file does not exist, will be created")

        s = RTIMU.Settings(self.SETTINGS_FILE)
        self.imu = RTIMU.RTIMU(s)

        logger.info("IMU Name: %s", self.imu.IMUName())

        if not self.imu.IMUInit():
            logger.error("IMU Init Failed")
            self.good = False
        else:
            logger.info("IMU Init Succeeded")
            self.good = True
            self.imu.setSlerpPower(0.02)
            #self.imu.setGyroEnable(True)
            self.imu.setAccelEnable(True)
            self.imu.setCompassEnable(True)

            self.poll_interval = self.imu.IMUGetPollInterval()
            logger.info("Recommended Poll Interval: %f", self.poll_interval)

    def __init__(self, settings_path="../dataset/RTIMULib"):
        self.SETTINGS_FILE = settings_path
        self.load_rtimu_lib()

    def get_headings(self):
        logger.info("about to get gyro reading")
        if self.good and self.imu.IMURead():
            logger.info("about to get data")
            # x, y, z = imu.getFusionData()
            # print("%f %f %f" % (x,y,z))
            data = self.imu.getIMUData()
            fusion_pose = data["fusionPose"]
            accel_pose= data["accel"]
            logger.info(accel_pose)
            pitch =  math.degrees(accel_pose[1])
            roll = math.degrees(accel_pose[0])
            yaw = math.degrees(accel_pose[2])

            #logger.info("pitch: %f roll: %f yaw: %f", pitch, roll, yaw)

            return pitch, roll, yaw
        else:
            return None, None, None


def main():

    accel = Gyro()

    while True:
        accel.get_headings()
        time.sleep(accel.poll_interval*1.0)


if __name__ == '__main__':
    main()
