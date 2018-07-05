import os
import random
import sys

# import matplotlib.image as mpimg

import numpy as np
import tensorflow as tf

from flask import Flask
from flask import request
from flask import jsonify

from . import rest_inference
from .utils import misc_utils as utils
from .utils import vocab_utils

app = Flask(__name__)

FLAGS = tf.contrib.training.HParams(
  attention='', attention_architecture='standard', avg_ckpts=False, batch_size=128, beam_width=0, check_special_token=True, ckpt='', colocate_gradients_with_ops=True, decay_scheme='', dev_prefix=None, dropout=0.2, embed_prefix=None, encoder_type='uni', eos='</s>', forget_bias=1.0, hparams_path=None, infer_batch_size=32, inference_input_file='/Users/mrzhao/learning/ai-translation/data/test.en', inference_list=None, inference_output_file='/Users/mrzhao/learning/ai-translation/data/output_infer', inference_ref_file=None, init_op='uniform', init_weight=0.1, jobid=0, learning_rate=1.0, length_penalty_weight=0.0, log_device_placement=False, max_gradient_norm=5.0, max_train=0, metrics='bleu', num_buckets=5, num_decoder_layers=None, num_embeddings_partitions=0, num_encoder_layers=None, num_gpus=1, num_inter_threads=0, num_intra_threads=0, num_keep_ckpts=5, num_layers=2, num_train_steps=12000, num_translations_per_input=1, num_units=32, num_workers=1, optimizer='sgd', out_dir='/Users/mrzhao/learning/nmt/nmt/translate_model', output_attention=True, override_loaded_hparams=False, pass_hidden_state=True, random_seed=None, residual=False, sampling_temperature=0.0, scope=None, share_vocab=False, sos='<s>', src=None, src_max_len=50, src_max_len_infer=None, steps_per_external_eval=None, steps_per_stats=100, subword_option='', test_prefix=None, tgt=None, tgt_max_len=50, tgt_max_len_infer=None, time_major=True, train_prefix=None, unit_type='lstm', vocab_prefix=None, warmup_scheme='t2t', warmup_steps=0
)
OUTPUT_DIR = '/Users/mrzhao/learning/nmt/nmt/translate_model'

def create_hparams(flags):
    """Create training hparams."""

    return tf.contrib.training.HParams(
        src=flags.src,
        tgt=flags.tgt,
        train_prefix=flags.train_prefix,
        dev_prefix=flags.dev_prefix,
        test_prefix=flags.test_prefix,
        vocab_prefix=flags.vocab_prefix,
        embed_prefix=flags.embed_prefix,
        out_dir=flags.out_dir,
        num_units=flags.num_units,
        num_layers=flags.num_layers,
        num_encoder_layers=flags.num_encoder_layers
            or flags.num_layers,
        num_decoder_layers=flags.num_decoder_layers
            or flags.num_layers,
        dropout=flags.dropout,
        unit_type=flags.unit_type,
        encoder_type=flags.encoder_type,
        residual=flags.residual,
        time_major=flags.time_major,
        num_embeddings_partitions=flags.num_embeddings_partitions,
        attention=flags.attention,
        attention_architecture=flags.attention_architecture,
        output_attention=flags.output_attention,
        pass_hidden_state=flags.pass_hidden_state,
        optimizer=flags.optimizer,
        num_train_steps=flags.num_train_steps,
        batch_size=flags.batch_size,
        init_op=flags.init_op,
        init_weight=flags.init_weight,
        max_gradient_norm=flags.max_gradient_norm,
        learning_rate=flags.learning_rate,
        warmup_steps=flags.warmup_steps,
        warmup_scheme=flags.warmup_scheme,
        decay_scheme=flags.decay_scheme,
        colocate_gradients_with_ops=flags.colocate_gradients_with_ops,
        num_buckets=flags.num_buckets,
        max_train=flags.max_train,
        src_max_len=flags.src_max_len,
        tgt_max_len=flags.tgt_max_len,
        src_max_len_infer=flags.src_max_len_infer,
        tgt_max_len_infer=flags.tgt_max_len_infer,
        infer_batch_size=flags.infer_batch_size,
        beam_width=flags.beam_width,
        length_penalty_weight=flags.length_penalty_weight,
        sampling_temperature=flags.sampling_temperature,
        num_translations_per_input=flags.num_translations_per_input,
        sos=(flags.sos if flags.sos else vocab_utils.SOS),
        eos=(flags.eos if flags.eos else vocab_utils.EOS),
        subword_option=flags.subword_option,
        check_special_token=flags.check_special_token,
        forget_bias=flags.forget_bias,
        num_gpus=flags.num_gpus,
        epoch_step=0,
        steps_per_stats=flags.steps_per_stats,
        steps_per_external_eval=flags.steps_per_external_eval,
        share_vocab=flags.share_vocab,
        metrics=flags.metrics.split(','),
        log_device_placement=flags.log_device_placement,
        random_seed=flags.random_seed,
        override_loaded_hparams=flags.override_loaded_hparams,
        num_keep_ckpts=flags.num_keep_ckpts,
        avg_ckpts=flags.avg_ckpts,
        num_intra_threads=flags.num_intra_threads,
        num_inter_threads=flags.num_inter_threads,
        )


def ensure_compatible_hparams(hparams, default_hparams, hparams_path):
    """Make sure the loaded hparams is compatible with new changes."""

    default_hparams = utils.maybe_parse_standard_hparams(default_hparams,
            hparams_path)

  # For compatible reason, if there are new fields in default_hparams,
  #   we add them to the current hparams

    default_config = default_hparams.values()
    config = hparams.values()
    for key in default_config:
        if key not in config:
            hparams.add_hparam(key, default_config[key])

  # Update all hparams' keys if override_loaded_hparams=True

    if default_hparams.override_loaded_hparams:
        for key in default_config:
            if getattr(hparams, key) != default_config[key]:
                utils.print_out('# Updating hparams.%s: %s -> %s'
                                % (key, str(getattr(hparams, key)),
                                str(default_config[key])))
                setattr(hparams, key, default_config[key])
    return hparams


def load_hparams(out_dir, default_hparams, hparams_path):
    """Load hparams from out_dir."""

    hparams = utils.load_hparams(out_dir)
    hparams = ensure_compatible_hparams(hparams, default_hparams,
            hparams_path)

    # Print HParams

    utils.print_hparams(hparams)
    return hparams


def predict(infer_data):
    default_hparams = create_hparams(FLAGS)
    inference_fn = rest_inference.inference

    # Load hparams.

    hparams = load_hparams(OUTPUT_DIR, default_hparams, FLAGS.hparams_path)

    # Inference indices

    hparams.inference_indices = None

    # Inference

    ckpt = FLAGS.ckpt
    if not ckpt:
        ckpt = tf.train.latest_checkpoint(OUTPUT_DIR)
    return inference_fn(ckpt, infer_data, hparams)


@app.route('/predict', methods=['POST'])
def main():
    infer_data = request.json
    translation = predict(infer_data)
    return jsonify(translation)

