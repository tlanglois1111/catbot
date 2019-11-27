"""camera_tf_trt.py

This is a Camera TensorFlow/TensorRT Object Detection code for
Jetson Nano.
"""

import time
import threading
import os

import logging
import logging.config

from cat_bot_trt import Gyro

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


def main():

    gyro = Gyro()
    gyro.start()

    while True:
        logger.info(gyro.get_headings())
        time.sleep(500)


if __name__ == '__main__':
    main()
