#!/usr/bin/env python

"""
Functions for the ensemble scenario
"""

import os
import os.path as osp
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import bisect

import json
import falcon
from falcon_multipart.middleware import MultipartMiddleware
from wsgiref import simple_server

import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2

from config import cfg




def propose_boxes_only(bm, model, tensors):
  # For each model, compute the proposals
  preprocessed_image = bm.preprocess(tf.to_float(tensors['image']))
  
  # Run the first stage only (proposals)
  (prediction_dict, proposal_boxes_normalized, proposal_scores, num_proposals) = \
      bm.propose_boxes_only(preprocessed_image, scope=model)
  
  # Returns them
  return {
    'prediction_dict': prediction_dict, 
    'proposal_boxes_normalized': proposal_boxes_normalized, 
    'proposal_scores': proposal_scores, 
    'num_proposals': num_proposals, 
  }

def merge_proposals(ensemble, proposals):
  scores = proposals[ensemble[0].name]['proposal_scores'].tolist()
  boxes = proposals[ensemble[0].name]['proposal_boxes_normalized'].tolist()

  for model in ensemble[1:]:
    for batch in range(len(scores)):
      for ix, s in enumerate(proposals[model.name]['proposal_scores'][batch]):
        if s > scores[batch][-1]:
          scores[batch] = scores[batch][::-1]
          index = len(scores[batch]) - bisect.bisect(scores[batch], s)
          scores[batch] = scores[batch][::-1]
          scores[batch].insert(index, s)
          boxes[batch].insert(index, proposals[model.name]['proposal_boxes_normalized'][batch][ix])
          scores[batch] = scores[batch][:-1]
          boxes[batch] = boxes[batch][:-1]
        else:
          break

  tmp_dict = proposals[ensemble[0].name]
  tmp_dict.update({'proposal_scores': np.array(scores), 
                   'proposal_boxes_normalized': np.array(boxes)})
  return tmp_dict


def classify_proposals_only(bm, model, tensors, isRfcn):
  # Run the second stage only and return the result
  detections = bm.classify_proposals_only(tensors['rpn_feat'], tensors['image_shape'], 
                                          tensors['prop_boxes'], tensors['num_prop'], 
                                          isRfcn, scope_prefix=model)
  
  # Restore lost parameters between stages
  tmp_dict = { 'image_shape': tensors['image_shape'] }
  tmp_dict.update(detections)
  detections = tmp_dict
  
  return detections

def merge_detections(ensemble, detections):
    new_classes_predictions = detections[ensemble[0].name]['class_predictions_with_background']
    new_refined_boxes = detections[ensemble[0].name]['refined_box_encodings']
    new_proposals_boxes = detections[ensemble[0].name]['proposal_boxes']
    for model in ensemble[1:]:
      new_classes_predictions += detections[model.name]['class_predictions_with_background']
      new_refined_boxes += detections[model.name]['refined_box_encodings']
      new_proposals_boxes += detections[model.name]['proposal_boxes']
    new_classes_predictions /= float(len(ensemble))
    new_refined_boxes /= float(len(ensemble))
    new_proposals_boxes /= float(len(ensemble))

    return {
      'class_predictions_with_background': new_classes_predictions, 
      'refined_box_encodings': new_refined_boxes, 
      'proposal_boxes': new_proposals_boxes, 
      'num_proposals': detections[ensemble[0].name]['num_proposals'], 
      'image_shape': detections[ensemble[0].name]['image_shape'], 
    }






