"""
Convolutional Neural Network for Fashion Landmarks Detection.
Daniel E  for COMP592 Project due 20th of April
"""

import os
import argparse
import tensorflow.compat.v1 as tf
from model import LandmarkModel

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--epochs', default=3, type=int, help='number of epochs')

cwd = os.getcwd()
train_filepath = os.path.join(cwd, "__TOY__\\train.tfrecords")
val_filepath = os.path.join(cwd, "__TOY__\\val.tfrecords")
export_path = cwd

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNEL = 3
output_size = 8 * 2


# Method defined for estimator argument(model_fn=...), returns specification for estimator depending on mode
# 'features' is the input tensor for the CNN model, which comes from tfrecords files as a image+landmark pair
# 'labels' are actual landmarks
def cnn_model_fn(features, labels, mode):
    """A model function implementing CNN regression for a custom Estimator."""

    init_model = LandmarkModel(output_size=output_size)  # triggers __init__ in model
    predictions = init_model(features)  # triggers __call__ to get model output, ie. predictions

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                "predict": tf.estimator.export.PredictOutput(predictions)
            })

    # Calculate loss using mean squared error.
    loss = tf.losses.mean_squared_error(labels, predictions)
    tf.identity(loss, name="loss")
    tf.summary.scalar("loss", loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimiser = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimiser.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
    else:
        train_op = None

    rmse_metrics = tf.metrics.root_mean_squared_error(labels, predictions)

    # Add the rmse to the collection of evaluation metrics.
    metrics = {"eval_mse": rmse_metrics}

    tf.identity(rmse_metrics[1], name="root_mean_squared_error")
    tf.summary.scalar("root_mean_squared_error", rmse_metrics[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics
    )


# Extracts data from serialised `tf.Example` protocol buffer, one at a time
def _parse_function(record):
    """"Mapping" function used by input_fn to unpack image and features from TFRecord file"""

    # Dictionary of data to be unpacked/extracted from given dataset
    feature_dict = {
        "image/filename": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "features/labels": tf.io.FixedLenFeature([16], tf.int64),
    }

    # Process one example from dataset in dictionary form
    parsed_features = tf.io.parse_single_example(record, feature_dict)

    # Decode the image part of the example and resize it
    decoded_image = tf.image.decode_image(parsed_features["image/encoded"])
    image_reshaped = tf.reshape(decoded_image, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])

    landmark = tf.cast(parsed_features["features/labels"], tf.float32)
    return image_reshaped, landmark


def input_fn(tfrecords, batch_size, num_epochs=None, shuffle=True):
    """Input function for Estimator"""

    # Add serialised dataset to variable
    dataset = tf.data.TFRecordDataset(tfrecords)

    # Build a pair of a feature dictionary and a label tensor for each example.
    dataset = dataset.map(_parse_function)
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    # Make dataset iterator
    iterator = tf.data.make_one_shot_iterator(dataset)

    # Return the feature and label for Estimator to process
    image, label = iterator.get_next()
    return image, label


def serving_input_receiver_fn():
    """An input function for TensorFlow Serving"""

    def _preprocess_image(image_bytes):
        image = tf.image.decode_jpeg(image_bytes, channels=IMG_CHANNEL)
        image.set_shape((None, None, IMG_CHANNEL))
        image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH],
                                       method=tf.image.ResizeMethod.BILINEAR,
                                       align_corners=False)
        return image

    # Placeholder for image to be encoded, for Tensorflow Serving
    image_bytes_list = tf.placeholder(shape=[None], dtype=tf.string, name="encoded_image_string_tensor")

    image = tf.map_fn(_preprocess_image, image_bytes_list, dtype=tf.float32, back_prop=False)

    return tf.estimator.export.TensorServingInputReceiver(
        features=image,
        receiver_tensors={"image_bytes": image_bytes_list})


'''
# ONCE local_init_op FINISHES, CHECKPOINT LISTENERS ARE CALLED AUTOMATICALLY 
def load_checkpoint_model(checkpoint_path, checkpoint_names):
    list_of_checkpoint_files = glob.glob(os.path.join(checkpoint_path, "*"))
    checkpoint_epoch_number = max([int(file.split(".")[1]) for file in list_of_checkpoint_files])
    checkpoint_epoch_path = os.path.join(checkpoint_path,
                                         checkpoint_names.format(epoch=checkpoint_epoch_number))
    resume_model = load_model(checkpoint_epoch_path)
    return resume_model, checkpoint_epoch_number
'''


def main(argv):
    """Builds, trains, and evaluates the model."""

    args = parser.parse_args(argv[1:])

    # Build a custom Estimator, using predefined model_fn.
    estimator = tf.estimator.Estimator(cnn_model_fn)

    print("Training...")
    estimator.train(input_fn=lambda: input_fn(train_filepath,
                                              batch_size=args.batch_size,
                                              num_epochs=args.epochs,
                                              shuffle=True), steps=10)
    print("Validating...")
    evaluation = estimator.evaluate(input_fn=lambda: input_fn(val_filepath,
                                                              batch_size=args.batch_size,
                                                              num_epochs=args.epochs,
                                                              shuffle=False))
    print(evaluation)

    # Export trained model as SavedModel.
    receiver_fn = serving_input_receiver_fn
    estimator.export_savedmodel(export_path, receiver_fn)


# Runs the program with an optional "main" function and "argv" list.
if __name__ == "__main__":
    # Change arg for set_verbosity to tf.logging.INFO for Estimator "INFO" logs
    tf.logging.set_verbosity(0)
    tf.app.run(main)
