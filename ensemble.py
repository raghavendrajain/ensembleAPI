#!/usr/bin/env python

"""
Class for an ensemble of model. WILL probably be very high level.
"""

import os.path as osp
import base64
from io import BytesIO
import numpy as np
from PIL import Image

import tensorflow as tf

from config import cfg
from utils import utils
from utils import ensemble_utils
from single_model import single_model


class ensemble:

  _session = None
  _inputs_postProcessing_tensors = {}
  _scenarios = {}
  
  def __init__(self, name, group, dataset_name):
    self._name = name
    self._dataset_name = dataset_name
    all_models = []
    for model in group:
      all_models.append(single_model(model, dataset_name))
    self._models_list = all_models


  @property
  def name(self):
    return self._name

  @property
  def models_list(self):
    return self._models_list

  @property
  def default_model(self):
    return self.models_list[0]

  @property
  def dataset_name(self):
    return self._dataset_name

  @property
  def nb_classes(self):
    return cfg.DATASET.NB_CATEGS[self._dataset_name]

  @property
  def session(self):
    return self._session

  @property
  def inputs_postProcess_tensors(self):
    return self._inputs_postProcessing_tensors

  @property
  def scenarios(self):
    return self._scenarios








  def hasRequiredFiles(self):
    for model in self.models_list:
      if not model.hasRequiredFilesForEnsemble():
        print "Missing files for model", model.name, "as part of an ensemble"
        return False
    return True

  def isRunnable(self):
    return False


  def getPostProcessingInputs(self):
    return {
        'class_predictions_with_background': tf.placeholder(tf.float32, shape=[300, self.nb_classes+1], name='class_predictions'), 
        'num_proposals':                     tf.placeholder(tf.int32, shape=[1, ], name='num_proposals'), 
        'image_shape':                       tf.placeholder(tf.int32, shape=[4, ], name='image_shape'), 
        'refined_box_encodings':             tf.placeholder(tf.float32, shape=[300, self.nb_classes, 4], name='refined_box_encodings'), 
        'proposal_boxes':                    tf.placeholder(tf.float32, shape=[1, 300, 4], name='proposal_boxes'), 
    }


  # /TODO\ only one ensemble at a time supported so far
  # An ensemble needs:
  # - A bunch of sessions for each model (first + second stage)
  # - A regular session for the post processing
  # - A bunch of inputs for each model (image, proposals, etc)
  # - A bunch of scenario for each model
  def prepare(self, all_models_list):
    for model in self.models_list:
      model.prepareForEnsemble()

    self._inputs_postProcessing_tensors = self.getPostProcessingInputs()

    all_scenarios = {}
    all_scenarios['proposals'] = {}
    all_scenarios['detections'] = {}

    common_variables = []
    for model in self.models_list:
      # Signature for the model
      scope_name = '-'.join([self.name, model.name])

      # First stage scenario
      all_scenarios['proposals'][model.name] = \
          ensemble_utils.propose_boxes_only(model.built, scope_name, 
                                            model.inputs_firstStage_tensors)
      
      # Second stage scenario
      all_scenarios['detections'][model.name] = \
          ensemble_utils.classify_proposals_only(model.built, scope_name, 
                                                 model.inputs_secondStage_tensors,
                                                 (model.name == 'rfcn'))
      
      # Create the session
      common_variables, variables_to_restore = \
          utils.get_variables(common_variables, tf.global_variables(), all_models_list, scope_name)
      model.init_session(variables_to_restore)
      
    # Postprocess scenario
    all_scenarios['post_process'] = self.default_model.built.postprocess(self.inputs_postProcess_tensors)

    # Globalize
    self._scenarios = all_scenarios

    # Regular session for post processing
    self._session = tf.Session('', graph=tf.get_default_graph())

  # Get the image in the ensemble format
  def get_image(self, raw_image):
    image = Image.open(BytesIO(base64.b64decode(raw_image)))
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    extended_image = np.expand_dims(image_np, axis=0)
    return extended_image

  # Predict
  def predict(self, image):

    # Run the first stage
    proposals = {}
    for model in self.models_list:
      inp = model.inputs_firstStage_tensors
      proposals[model.name] = model.session.run(
          self.scenarios['proposals'][model.name], 
          feed_dict={inp['image']: image}
      )

    # For each image, merge the results of all the models
    ensembled_proposals = ensemble_utils.merge_proposals(self.models_list, proposals)

    # Run the second stage
    detections = {}
    for model in self.models_list:
      inp = model.inputs_secondStage_tensors
      detections[model.name] = model.session.run(
        self.scenarios['detections'][model.name], 
        feed_dict={inp['rpn_feat']:    proposals[model.name]['prediction_dict']['rpn_features_to_crop'],
                   inp['image_shape']: proposals[model.name]['prediction_dict']['image_shape'],
                   inp['prop_boxes']:  ensembled_proposals['proposal_boxes_normalized'],
                   inp['num_prop']:    ensembled_proposals['num_proposals']}
      )

    # For each image, merge the results of all the models
    ensembled_detections = ensemble_utils.merge_detections(self.models_list, detections)

    # Postprocess
    inp = self.inputs_postProcess_tensors
    postprocessed_detections = self.session.run(
        self.scenarios['post_process'], 
        feed_dict={inp['class_predictions_with_background']: ensembled_detections['class_predictions_with_background'],
                   inp['refined_box_encodings']:             ensembled_detections['refined_box_encodings'],
                   inp['proposal_boxes']:                    ensembled_detections['proposal_boxes'],
                   inp['num_proposals']:                     ensembled_detections['num_proposals'], 
                   inp['image_shape']:                       ensembled_detections['image_shape']}
      )

    num_detections = postprocessed_detections['num_detections']
    scores = postprocessed_detections['detection_scores']
    boxes = postprocessed_detections['detection_boxes']
    classes = postprocessed_detections['detection_classes']

    return boxes, scores, classes, num_detections















