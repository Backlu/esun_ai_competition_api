#!/usr/bin/env python3
# -*- coding: utf-8 -*

import datetime, time, cv2, base64, hashlib, uvicorn
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
from PIL import Image
from json import JSONDecodeError
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.logger import logger as fastapi_logger
import logging
from logging.handlers import RotatingFileHandler
from .api_scheme import ModelOutput, HealthCheckOutput
from ..src.util_v1 import CWordRecog

# --- init ---
tfmodel = CWordRecog()
CAPTAIN_EMAIL = 'k23038988@gmail.com'
SALT = 'my_salt'
tags_metadata = [{"name"       : "Model",
                  "description": "Operations for the lightgbm model",
                  },
                 {"name"       : "Default",
                  "description": "Basic function for API"
                  }
                 ]


#--- AI Inference Functions ---
def generate_server_uuid(input_string: str):
    """generate_server_uuid Create server_uuid.

    Args:
        input_string (str): Information to be encoded as server_uuid

    Returns:
        str: unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def _check_datatype_to_string(prediction) -> bool:
    """_check_datatype_to_string Check if your prediction is `str`.

    Args:
        prediction (any): Model Prediction.

    Raises:
        error.: True or raise TypeError.
    """
    if isinstance(prediction, str):
        return True
    raise TypeError('Prediction is not in string type.')


def predict(image: np.ndarray) -> str:
    """predict Predict your model result.

    Args:
        image (np.ndarray): Input OCR Image.

    Returns:
        str: Word
    """
    # - PUT YOUR MODEL INFERENCING CODE HERE -
    s = time.time()
    pred = tfmodel(image)
    word, score = pred['word'],pred['score']
    prediction = word
    e = time.time()
    print(f'---- pred: {prediction}, score: {score:.3f} ----, --- time ---:{e-s}')
    if _check_datatype_to_string(prediction):
        return prediction
    return 'isnull'


#--- FastAPI Functions ---
app = FastAPI(title='ESUN OCR Detection',
              description='AA',
              version='1.0.0',
              openapi_tags=tags_metadata)


@app.get('/health', response_model=HealthCheckOutput, tags=["Health"])
async def health_check():
    return {"health": "True"}


def save_input_image(image: np.ndarray, predict_ans, time_duration):
    im=Image.fromarray(image)
    now_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    im.save(f'ocr_detection/log/image/data{now_ts}.jpeg')
    fastapi_logger.info(f'FILE: [data{now_ts}.jpeg] PRED: [{predict_ans}] DURATION: [{time_duration}]')


def base64_to_binary_for_cv2(image_64_encoded):
    """base64_to_binary_for_cv2 Convert base64 to numpy.ndarray for cv2.

    Args:
        image_64_encoded (str): Image that encoded in base64 string format.

    Returns:
        np.ndarray: An image.
    """
    img_base64_binary = image_64_encoded.encode("utf-8")
    img_binary = base64.b64decode(img_base64_binary)
    image = cv2.imdecode(np.frombuffer(img_binary, np.uint8), cv2.IMREAD_COLOR)
    return image


@app.post('/inference')
async def inference(inference_json: Request, background_tasks: BackgroundTasks, tags=["Model"]):
    """inference API that return your model predictions when E.SUN calls this API.
    """
    # Input data
    try:
        data_json = await inference_json.json()
        message = "Success"
    except JSONDecodeError:
        data_json = None
        message = "Received data is not a valid JSON"
    fastapi_logger.info(message)
    t_start = time.time()
    # Transform Image from bunary to nparray
    image_64_encoded = data_json['image']
    image = base64_to_binary_for_cv2(image_64_encoded)

    # Predict
    answer = predict(image)
    ts = int(datetime.datetime.now().utcnow().timestamp())
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + str(ts))
    
    t_end = time.time()
    time_duration = t_end-t_start
    background_tasks.add_task(save_input_image, image, answer, time_duration)

    return {
            "esun_uuid"       : data_json['esun_uuid'],
            "server_uuid"     : server_uuid,
            "server_timestamp": ts,
            "answer"          : answer,
            }

#--- main function ---
if __name__ == "__main__":
    arg_parser = ArgumentParser(usage='Usage: python ' + __file__ + ' [--port <port>] [--help]')
    arg_parser.add_argument('-p', '--port', default=5000, help='port')
    options = arg_parser.parse_args()
    
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(levelname)s [%(thread)d] - %(message)s", "%Y-%m-%d %H:%M:%S")
    handler = RotatingFileHandler('log/ocr.log', backupCount=0)
    logging.getLogger().setLevel(logging.NOTSET)
    fastapi_logger.addHandler(handler)
    handler.setFormatter(formatter)
    fastapi_logger.info('****************** Starting Server *****************')
    
    config = {}
    config['log_config'] = {'version': 1,
                            'disable_existing_loggers': False,
                            'formatters': {'default': {'()': 'uvicorn.logging.DefaultFormatter', 'fmt': '%(levelprefix)s %(message)s', 'use_colors': None},
                           'access': {'()': 'uvicorn.logging.AccessFormatter', 'fmt': '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'}},
            'handlers': {'default': {'formatter': 'default', 'class': 'logging.StreamHandler', 'stream': 'ext://sys.stderr'},
                         'access': {'formatter': 'access', 'class': 'logging.StreamHandler', 'stream': 'ext://sys.stdout'}},
            'loggers': {'uvicorn': {'handlers': ['default'], 'level': 'INFO'},
                        'uvicorn.error': {'level': 'INFO', 'handlers': ['default'], 'propagate': True},
                        'uvicorn.access': {'handlers': ['access'], 'level': 'INFO', 'propagate': False},
                        },
        }
    config['log_config']['loggers']['quart'] = {'handlers': ['default'], 'level': 'INFO'}

    # Run API app
    uvicorn.run(app, host="0.0.0.0", port=int(options.port), **config)
