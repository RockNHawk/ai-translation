import os, argparse

import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(model_dir, meta_name, export_base_path, version_number):
    export_path = os.path.join(
        tf.compat.as_bytes(export_base_path),
        tf.compat.as_bytes(str(version_number))
    )
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_name))
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    saver.restore(sess, latest_ckpt)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    feature_configs = {
        'x': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'y': tf.FixedLenFeature(shape=[], dtype=tf.string)
    }
    serialized_example = tf.placeholder(tf.string, name="tf_example")
    tf_example = tf.parse_example(serialized_example, feature_configs)
    x = tf.identity(tf_example['x'], name='x')
    y = tf.identity(tf_example['y'], name='y')
    predict_input = x
    predict_output = y
    predict_signature_def_map = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={
            tf.saved_model.signature_constants.PREDICT_INPUTS: predict_input
        },
        outputs={
            tf.saved_model.signature_constants.PREDICT_OUTPUTS: predict_output
        }
    )

    legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: predict_signature_def_map
        },
        legacy_init_op=legacy_init_op,
        assets_collection=None
    )
    builder.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
    parser.add_argument("--meta_name", type=str, default="", help="The name of the mata file")
    parser.add_argument("--export_path", type=str, default="", help="The folder to export freezed model")
    parser.add_argument("--version", type=str, default="", help="The version of freezed model")
    args = parser.parse_args()

    freeze_graph(args.model_dir, args.meta_name, args.export_path, args.version)



