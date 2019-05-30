import argparse
import base64
import os
import time

import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify

from mattnet import MattNet

app = Flask(__name__)

mattnet = None

CACHE_DIR = "../cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Image processing snippets
def imread(image_path):
    """Provides a wrapper over cv2.imread that converts to RGB space
    """
    bgr_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img

def img_to_b64(img):
    """Converts RGB -> BGR, encodes to jpeg and then converts to base64
    """
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, nparr = cv2.imencode('.png', bgr_img)
    b64_img_str = base64.b64encode(nparr).decode()
    return b64_img_str


def b64_to_img(b64_img_str):
    """Converts base64 string back to numpy array
    Assume encoded in BGR format by the above function
    """
    buf = base64.b64decode(b64_img_str)
    nparr = np.frombuffer(buf, np.uint8)
    bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img



@app.route("/detect", methods=["POST"])
def detect():
    # arguments
    args = request.json or request.form
    expr = args['expr'] # natural language
    b64_img_str = args['b64_img_str'] # image

    print('expr', expr)
    time_stamp = str(int(time.time()))

    # Save image to image path
    img = b64_to_img(b64_img_str)
    img_path = os.path.join(CACHE_DIR, "{}_{}.png".format(expr, time_stamp))
    cv2.imwrite(img_path, img)

    # Inference
    img_data = mattnet.forward_image(img_path)
    entry = mattnet.comprehend(img_data, expr)

    # Convert binary mask to image mask(255)
    pred_mask = entry['pred_mask']
    pred_mask_img = pred_mask * 255

    mask_img_str = img_to_b64(pred_mask_img)

    obj = {}
    obj["mask_img_str"] = mask_img_str
    return jsonify(obj)

def test_one_image():

    IMAGE_DIR = '../pyutils/mask-faster-rcnn/data/coco/train2014'
    img_path = os.path.join(IMAGE_DIR, 'COCO_train2014_'+str(229598).zfill(12)+'.jpg')
    print('img_path', img_path)
    img_data = mattnet.forward_image(img_path)

    expr = 'man in black'
    entry = mattnet.comprehend(img_data, expr)

    print(entry)
    pred_mask = entry['pred_mask']
    assert pred_mask.dtype == np.uint8
    cv2.imwrite("mask.png", pred_mask * 255)


def request_one_image():
    IMAGE_DIR = '../pyutils/mask-faster-rcnn/data/coco/train2014'
    img_path = os.path.join(IMAGE_DIR, 'COCO_train2014_'+str(229598).zfill(12)+'.jpg')

    img = imread(img_path)

    expr = "man in black"
    b64_img_str = img_to_b64(img)
    data = {
            "expr": expr,
            "b64_img_str": b64_img_str
            }

    url = "http://0.0.0.0:6000/detect"

    res = requests.post(url, data=data)
    print(res.status_code)
    if res.status_code == 200:
        mask_img_str = res.json()['mask_img_str']
        pred_mask = b64_to_img(mask_img_str)
        print("shape", pred_mask.shape)
        print('nonzero', np.count_nonzero(pred_mask))
        cv2.imwrite("test_request.png", pred_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # MAttNet Args
    parser.add_argument('--dataset', type=str, default="refcoco")
    parser.add_argument('--splitBy', type=str, default="unc")
    parser.add_argument('--model_id', type=str, default="mrcn_cmr_with_st")

    # Server Args
    parser.add_argument('--mode', type=int, default=1, help="1: test_one_image, 2: server")
    parser.add_argument('--port', type=str, default=6000)
    parser.add_argument('-d', '--debug', action="store_true")
    args = parser.parse_args()


    mode = args.mode
    if mode < 3:
        mattnet = MattNet(args)

    # Depends on mode
    if mode == 1:
        print("Test one image")
        test_one_image()
    elif mode == 2:
        print("Serve model on port: {}".format(args.port))
        app.run(host="0.0.0.0",
                port=args.port,
                debug=args.debug)
    elif mode == 3:
        print("Request one image")
        request_one_image()
    else:
        raise ValueError("Unknown mode: {}".format(mode))



