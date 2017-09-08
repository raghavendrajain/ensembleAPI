#!/usr/bin/env python

"""
Runner for the local API, will run the API on http://localhost:8000
$ python api.py
"""

import json
import falcon
from falcon_multipart.middleware import MultipartMiddleware
from wsgiref import simple_server

from ensemble import ensemble
from single_model import single_model
from config import cfg
from utils import utils



# Ensembles to deal with
ensembles_list = [ 
    ensemble('ensemble', ['frcnn', 'inception', 'rfcn'], 'foodinc'),
]
single_models_list = [ 
    single_model('frcnn',      'foodinc'), 
    single_model('frcnn_coco', 'ms_coco'), 
    single_model('inception',  'foodinc'), 
    single_model('rfcn',       'foodinc'), 
    single_model('mobilenet',  'foodinc'), 
    single_model('uecfood',    'uecfood_256'), 
]



# API
class FoodImageRecognitionApi(object):

  def on_get(self, req, resp):
    # Parse the request
    doc = json.load(req.bounded_stream)
    model_name = doc['model']      or ''
    raw_image =  doc['image_data'] or ''


    # High level, to do: deal with multiple ensemble (could also remove the if)
    if model_name.startswith('ensemble'):
      # The actual ensemble
      ensemble_to_use = ensembles_dict[model_name]

      # The input image
      readable_image = ensemble_to_use.get_image(raw_image)

      # Predict
      boxes, scores, classes, num_detections = \
          ensemble_to_use.predict(readable_image)
    else:
      # The actual model
      model_to_use = single_models_dict[model_name]

      # The input image
      readable_image = model_to_use.get_image(raw_image)

      # Predict
      boxes, scores, classes, num_detections = \
          model_to_use.predict(readable_image)
    
    # Parse the result into a json
    doc = {
      'boxes': boxes.tolist(), 
      'scores': scores.tolist(), 
      'classes': classes.tolist(), 
      'num_detections': num_detections.tolist(), 
    }

    # Create a JSON representation of the resource
    resp.body = json.dumps(doc, ensure_ascii=False)

    # Worked well
    resp.status = falcon.HTTP_200


# List valid models
class DatasetsInfos(object):
  def on_get(self, req, resp):
    # Parse the request
    doc = json.load(req.bounded_stream)
    model_name = doc['model']      or ''

    if model_name.startswith('ensemble'):
      dataset = ensembles_dict[model_name].dataset_name
    else:
      dataset = single_models_dict[model_name].dataset_name

    resp.body = json.dumps(dataset, ensure_ascii=False)
    resp.status = falcon.HTTP_200


# List valid models
class ValidModelsApi(object):
  def on_get(self, req, resp):
    models_list = [e.name for e in ensembles_list] + [m.name for m in single_models_list]
    resp.body = json.dumps(models_list, ensure_ascii=False)
    resp.status = falcon.HTTP_200



if __name__ == '__main__':

    # Assert all the models have a unique ID (for common weights sharing)
    assert utils.allUnique([x.name for x in single_models_list + ensembles_list]), \
      'All the models (single and ensemble) should have a unique ID (name)'

    # Check which of the ensembles can be used
    for ix, ensemble_element in reversed(list(enumerate(ensembles_list))):
      if not ensemble_element.hasRequiredFiles():
        print 'Missing information for ensemble', ensemble_element.name
        print '-> won\'t be served'
        ensembles_list.pop(ix)


    # Check which of the single model can be used
    for ix, single_model_element in reversed(list(enumerate(single_models_list))):
      if not single_model_element.hasRequiredFiles():
        print 'Missing information for model', single_model_element.name
        print '-> won\'t be served'
        single_models_list.pop(ix)

    # Names of all the models
    all_models_names = []
    for e in ensembles_list:
      all_models_names += ['-'.join([e.name, m.name]) for m in e.models_list]
    all_models_names += [m.name for m in single_models_list]

    # Dictionaries for easy access
    ensembles_dict = {}
    single_models_dict = {}
    for ensemble_element in ensembles_list:
      ensembles_dict[ensemble_element.name] = ensemble_element
    for single_model_element in single_models_list:
      single_models_dict[single_model_element.name] = single_model_element

    # Load the weights for the ensemble
    for ensemble_element in ensembles_list:
      ensemble_element.prepare(all_models_names)

    # Load the weights for the single models
    for single_model_element in single_models_list:
      single_model_element.prepare()
    


    # Creates the API
    app = falcon.API(middleware=[MultipartMiddleware()])
    app.add_route('/' + cfg.API.ROUTE, FoodImageRecognitionApi())
    app.add_route('/' + cfg.API.LIST, ValidModelsApi())
    app.add_route('/' + cfg.API.DATASET, DatasetsInfos())

    # Host it in a simple local server
    httpd = simple_server.make_server('', cfg.API.PORT, app)

    # Serve forever
    print "Serving on " + cfg.API.ADDRESS + "..."
    print "Proposed models:"
    models_list = [e.name for e in ensembles_list] + [m.name for m in single_models_list]
    print '\n'.join(['- ' + m for m in models_list])
    httpd.serve_forever()


