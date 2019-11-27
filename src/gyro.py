"""camera_tf_trt.py

This is a Camera TensorFlow/TensorRT Object Detection code for
Jetson Nano.
"""

import time
import threading
import os

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
        'handlers': ['console'],
        'level': 'WARN',
        'propagate': False
    }
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)


class Gyro(threading.Thread):
    """RTIMU encapsulates use of gyro device"""

    def _load_rtimu_lib(self):

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
            self.imu.setGyroEnable(True)
            self.imu.setAccelEnable(True)
            self.imu.setCompassEnable(True)

            self.poll_interval = self.imu.IMUGetPollInterval()
            logger.info("Recommended Poll Interval: %f", self.poll_interval)

    def __init__(self, settings_path="../dataset/RTIMULib"):
        threading.Thread.__init__(self)
        self.SETTINGS_FILE = settings_path
        self._load_rtimu_lib()
        self.keep_running = True
        self.data = []

    def run(self):
        logger.info("running gyro")
        while self.keep_running:
            if self.good and self.imu.IMURead():
                self.data = self.imu.getIMUData()
            time.sleep(self.poll_interval*1.0/1000.0)

    def get_headings(self):
        logger.info("about to get gyro reading")
        if len(self.data) > 0:
            return self.data["accel"]
        else:
            return self.data

    def stop(self):
        self.keep_running = False



def main():

    accel = Gyro()
    accel.start()

    while True:
        logger.info(accel.get_headings())
        time.sleep(2000)


if __name__ == '__main__':
    main()
