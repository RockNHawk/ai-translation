#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""To perform inference on test set given a trained model."""

from __future__ import print_function

import codecs
import time
import numpy as np

import tensorflow as tf

from  import attention_model
from  import gnmt_model
from  import model as nmt_model
from  import model_helper
from utils import misc_utils as utils
from utils import evaluation_utils

__all__ = ['inference', 'single_worker_inference', 'decode_and_evaluate'
           , 'get_translation']


def inference(
    ckpt,
    inference_data,
    hparams,
    num_workers=1,
    jobid=0,
    scope=None,
    ):
    """Perform translation."""

    if not hparams.attention:
        model_creator = nmt_model.Model
    elif hparams.attention_architecture == 'standard':
        model_creator = attention_model.AttentionModel
    elif hparams.attention_architecture in ['gnmt', 'gnmt_v2']:
        model_creator = gnmt_model.GNMTModel
    else:
        raise ValueError('Unknown model architecture')
    infer_model = model_helper.create_infer_model(model_creator,
            hparams, scope)

    return single_worker_inference(infer_model, ckpt, inference_data,
                                   hparams)


def single_worker_inference(
    infer_model,
    ckpt,
    inference_data,
    hparams,
    ):
    """Inference with a single worker."""

    with tf.Session(graph=infer_model.graph,
                    config=utils.get_config_proto()) as sess:
        loaded_infer_model = model_helper.load_model(infer_model.model,
                ckpt, sess, 'infer')
        sess.run(infer_model.iterator.initializer,
                 feed_dict={infer_model.src_placeholder: inference_data,
                 infer_model.batch_size_placeholder: hparams.infer_batch_size})

    # Decode

        utils.print_out('# Start decoding')
        return decode(
            'infer',
            loaded_infer_model,
            sess,
            ref_file=None,
            metrics=hparams.metrics,
            subword_option=hparams.subword_option,
            beam_width=hparams.beam_width,
            tgt_eos=hparams.eos,
            num_translations_per_input=hparams.num_translations_per_input,
            )


def decode(
    name,
    model,
    sess,
    ref_file,
    metrics,
    subword_option,
    beam_width,
    tgt_eos,
    num_translations_per_input=1,
    ):
    """Decode a test set."""

  # Decode

    utils.print_out('  decoding.')
    translations = []

    start_time = time.time()
    num_sentences = 0

    num_translations_per_input = max(min(num_translations_per_input,
            beam_width), 1)
    while True:
        try:
            (nmt_outputs, _) = model.decode(sess)
            if beam_width == 0:
                nmt_outputs = np.expand_dims(nmt_outputs, 0)

            batch_size = nmt_outputs.shape[1]
            num_sentences += batch_size

            for sent_id in range(batch_size):
                for beam_id in range(num_translations_per_input):
                    translation = get_translation(nmt_outputs[beam_id],
                            sent_id, tgt_eos=tgt_eos,
                            subword_option=subword_option)

                    translations.append(translation)
        except tf.errors.OutOfRangeError:
            utils.print_time('  done, num sentences %d, num translations per input %d'
                              % (num_sentences,
                             num_translations_per_input), start_time)
            break

    return translations


def get_translation(
    nmt_outputs,
    sent_id,
    tgt_eos,
    subword_option,
    ):
    """Given batch decoding outputs, select a sentence and turn to text."""

    if tgt_eos:
        tgt_eos = tgt_eos.encode('utf-8')

  # Select a sentence

    output = nmt_outputs[sent_id, :].tolist()

  # If there is an eos symbol in outputs, cut them at that point.

    if tgt_eos and tgt_eos in output:
        output = output[:output.index(tgt_eos)]

    if subword_option == 'bpe':  # BPE
        translation = utils.format_bpe_text(output)
    elif subword_option == 'spm':  # SPM
        translation = utils.format_spm_text(output)
    else:
        translation = utils.format_text(output)

    return translation

