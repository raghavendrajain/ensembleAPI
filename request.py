#!/usr/bin/env python

"""
Send a request to the API 
$ python request.py --image path/to/image.jpg \
                    --model Faster_RCNN_ResNet101_Foodinc_950k
"""

import os
import argparse

import json
import requests

import base64
import cStringIO

from PIL import Image

from config import cfg


def getArguments():
    """Defines and parses command-line arguments to this script."""
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument('--image', default=cfg.TEST.LOCAL_IMAGE_PATH, help='\
    The image to send.')
    parser.add_argument('--model', default='frcnn', help='\
    The model to use.')

    return parser.parse_args()


# Default display for a detection
def printDetections(boxes, scores, classes, num_detections):
    for i in range(num_detections):
        print boxes[0][i][0], \
              boxes[0][i][1], \
              boxes[0][i][2], \
              boxes[0][i][3], \
              classes[0][i], \
              scores[0][i]


if __name__ == '__main__':

    # Get the arguments
    args = getArguments()
    
    # Check conditions
    if not os.path.isfile(args.image):
        print "Can't find image", args.image
        raise SystemExit
    
    # Read and encode the image
    image = Image.open(args.image)
    buffer = cStringIO.StringIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue())
    
    # Get the arguments
    arguments = {
        'model': args.model,
        'image_data': img_str,
    }

    # Send the request
    r = requests.get(cfg.API.ADDRESS_GET, data=json.dumps(arguments))
    
    # Display the results
    print r.text
    print len(json.loads(r.text)), 'detections found'


