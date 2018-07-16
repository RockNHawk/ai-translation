import os
import random
import sys

# import matplotlib.image as mpimg

import numpy as np
import tensorflow as tf

from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS

from . import rest_inference
from .utils import misc_utils as utils
from .utils import vocab_utils
from . import model as nmt_model
from . import attention_model
from . import gnmt_model
from . import model_helper
import argparse
from . import nmt


app = Flask(__name__)
app.config['JSON_AS_ASCII']=False
cors = CORS(app)

OUTPUT_DIR = os.environ['OUTPUT_DIR']
inference_fn = rest_inference.single_worker_inference

# Load hparams.
nmt_parser = argparse.ArgumentParser()
nmt.add_arguments(nmt_parser)
FLAGS, unparsed = nmt_parser.parse_known_args()
default_hparams = nmt.create_hparams(FLAGS)
hparams = nmt.create_or_load_hparams(OUTPUT_DIR, default_hparams, FLAGS.hparams_path,False)

# Inference indices

hparams.inference_indices = None

# Inference

ckpt = FLAGS.ckpt
if not ckpt:
    ckpt = tf.train.latest_checkpoint(OUTPUT_DIR)

if not hparams.attention:
    model_creator = nmt_model.Model
elif hparams.attention_architecture == 'standard':
    model_creator = attention_model.AttentionModel
elif hparams.attention_architecture in ['gnmt', 'gnmt_v2']:
    model_creator = gnmt_model.GNMTModel
else:
    raise ValueError('Unknown model architecture')
infer_model = model_helper.create_infer_model(model_creator,
            hparams, None)

def predict(infer_data):
    return inference_fn(infer_model, ckpt, infer_data,
                                   hparams)

@app.route('/index')
def index():
    return 'hello, ai-translation'
    
@app.route('/predict', methods=['POST'])
def main():
    infer_data = request.get_json()
    translation = predict(infer_data)
    return jsonify(translation)

