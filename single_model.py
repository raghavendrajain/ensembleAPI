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

from google.protobuf import text_format
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2

from config import cfg
from utils import ensemble_utils


class single_model:

  _built_model = None
  _inputs_firstStage_tensors = {}
  _inputs_secondStage_tensors = {}
  _detection_graph = None
  _session = None

  def __init__(self, name, dataset_name):
    self._name = name
    self._dataset_name = dataset_name


  @property
  def name(self):
    return self._name

  @property
  def dataset_name(self):
    return self._dataset_name

  @property
  def config_file(self):
    return osp.join(cfg.CONFIGS_DIR, self.name + '.config')

  @property
  def model_file(self):
    return osp.join(cfg.MODELS_DIR, self.name + '.pb')

  @property
  def latest_ckpt_file(self):
    path_to_ckpt = osp.join(cfg.MODELS_DIR, self.name)
    return tf.train.latest_checkpoint(path_to_ckpt)
  
  @property
  def built(self):
    return self._built_model

  @property
  def inputs_firstStage_tensors(self):
    return self._inputs_firstStage_tensors

  @property
  def inputs_secondStage_tensors(self):
    return self._inputs_secondStage_tensors

  @property
  def detection_graph(self):
    return self._detection_graph

  @property
  def session(self):
    return self._session






  def hasRequiredFiles(self):
    return osp.isfile(self.model_file) 

  def hasRequiredFilesForEnsemble(self):
    return osp.isfile(self.config_file) and self.latest_ckpt_file

  def isRunnable(self):
    return False





  # Builder for the model
  def get_built_model(self):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(self.config_file, 'r') as f:
      text_format.Merge(f.read(), pipeline_config)
    model_config = pipeline_config.model
    return model_builder.build(model_config, False)


  # Get the inputs tensor
  def get_inputs(self):
    first_stage = {}
    second_stage = {}

    # Input for the first stage: the image
    first_stage['image'] = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='image_tensor')

    # Input for the second stage: the selected proposals
    rpn_shape = cfg.MODEL.RPN_SHAPE[self.name]
    second_stage['rpn_feat']    = tf.placeholder(tf.float32, shape=rpn_shape, name='rpn_feat')
    second_stage['prop_boxes']  = tf.placeholder(tf.float32, shape=[1, 300, 4], name='prop_boxes')
    second_stage['num_prop']    = tf.placeholder(tf.int32, shape=[1], name='num_prop')
    second_stage['image_shape'] = tf.placeholder(tf.int32, shape=[4], name='img_shape')

    return first_stage, second_stage


  # Prepare for the single model: get the default graph
  def prepare(self):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(self.model_file, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    detection_graph.as_default()
    self._detection_graph = detection_graph
    self._session = tf.Session(graph=detection_graph)


  # Get the image in the model format
  def get_image(self, raw_image):
    image = Image.open(BytesIO(base64.b64decode(raw_image)))
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    return image_np


  # 
  def predict(self, image):
    image_np_expanded = np.expand_dims(image, axis=0)
    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
    classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
    (boxes, scores, classes, num_detections) = self.session.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # General post processing
    label_id_offset = 1
    num_detections = np.squeeze(num_detections, axis=0)
    scores = np.squeeze(scores, axis=0)
    boxes = np.squeeze(boxes, axis=0)
    classes = np.squeeze(classes, axis=0) + label_id_offset

    return boxes, scores, classes, num_detections


  # Prepare for the ensemble: get the built model and the inputs
  def prepareForEnsemble(self):
    self._built_model = self.get_built_model()
    
    # Input for the first stage: the image
    self._inputs_firstStage_tensors, self._inputs_secondStage_tensors = \
        self.get_inputs()


  # Helper for the session
  def init_session(self, variables_to_restore):
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session('', graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, self.latest_ckpt_file)
    self._session = sess






