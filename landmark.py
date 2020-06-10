'''
Convolutional Neural Network for Fashion Landmarks Detection.
Daniel E 201190945 for COMP592 Project due 20th of April
Download tfrecords files from https://1drv.ms/u/s!Au5dCZ4YubPGg4giMoXai4H2lLnBRA?e=fHdS9u
'''
import sys
import numpy as np
import tensorflow.compat.v1 as tf
from model import LandmarkModel

train_filepath = "C:\\Users\\Nilan Delancourt\\Desktop\projectenv\\dataset\\train.tfrecords"
val_filepath = "C:\\Users\\Nilan Delancourt\\Desktop\\projectenv\\dataset\\val.tfrecords"
log_path = "C:\\Users\\Nilan Delancourt\\Desktop\\projectenv\\logs"
export_path = "C:\\Users\\Nilan Delancourt\\Desktop\\projectenv\\"
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNEL = 3
output_size = 8*2

#funct below goes into estimator(model_fn=...), returns specification for estiamtor depending on mode
def cnn_model_fn(features, labels, mode):
    #features is the input tensor for the cnn model in model.py, which comes from the tfrecords file as a image+landmark pair
    #labels are actual landmarks, predictions (below are predicted landmarks), both dict or tf.tensor
    init_model = LandmarkModel(output_size=output_size) # trigger __init__
    predictions = init_model(features) #triggers LandmarkModel.__call__ to get model output, ie. predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                "predict": tf.estimator.export.PredictOutput(predictions)
            })
    #if mode==PREDICT, the return above essentially ends exec in this function, and execution continues
    #without the code below being reached. 
    
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
    
################################################################################################
'''
unpacks/ Takes features and image from TFRecord file and returns in different fromat
list_of_tfrecord_files = [dir1, dir2, dir3, dir4]
dataset = tf.data.TFRecordDataset(list_of_tfrecord_files)
iterator thru list, return image label pair for training. 
PROBLEM IS : WHEN RESIZING LANDMARKS BECOME INVALID
'''
def _parse_function(record):
    # Extract data from a `tf.Example` protocol buffer.######################################
    # Defaults are not specified since both keys are required.
    feature_dict = {
        "image/filename": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "features/labels": tf.io.FixedLenFeature([16], tf.int64),
    }
    # serialized = record(seri), features = feature_dict(dict)
    # (foreach image (dataset.map)) unpack examplerow according to dict.
    parsed_features = tf.io.parse_single_example(record, feature_dict)

    # (foreach image) Extract features from single example
    # decode image and resize and make landmarks to floats. Return the 2
    decoded_image = tf.image.decode_image(parsed_features["image/encoded"])
    image_reshaped = tf.reshape(decoded_image, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
        
    landmark = tf.cast(parsed_features["features/labels"], tf.float32)
    return image_reshaped, landmark

#################################################################################################
def input_fn(tfrecords, batch_size, num_epochs=None, shuffle=True):
    #.A Dataset comprising records from one or more TFRecord files.
    dataset = tf.data.TFRecordDataset(tfrecords)

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(_parse_function)
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)


    # Make dataset iterator. DEPRECATED IN 2.0
    iterator = tf.data.make_one_shot_iterator(dataset)

    # Return the feature and label.
    image, label = iterator.get_next()
    return image, label
    
######################################################################################################
def serving_input_receiver_fn():
    '''An input function for TensorFlow Serving.'''
    def _preprocess_image(image_bytes):
        image = tf.image.decode_jpeg(image_bytes, channels=IMG_CHANNEL)
        image.set_shape((None, None, IMG_CHANNEL))
        image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH],
                                       method=tf.image.ResizeMethod.BILINEAR,
                                       align_corners=False)
        return image

    image_bytes_list = tf.placeholder(shape=[None], dtype=tf.string, name="encoded_image_string_tensor")
    image = tf.map_fn(_preprocess_image, image_bytes_list,
                      dtype=tf.float32, back_prop=False)

    return tf.estimator.export.TensorServingInputReceiver(
        features=image,
        receiver_tensors={"image_bytes": image_bytes_list})

 #######################################################################################################    
def load_checkpoint_model(checkpoint_path, checkpoint_names):
    list_of_checkpoint_files = glob.glob(os.path.join(checkpoint_path, "*"))
    checkpoint_epoch_number = max([int(file.split(".")[1]) for file in list_of_checkpoint_files])
    checkpoint_epoch_path = os.path.join(checkpoint_path,
                                         checkpoint_names.format(epoch=checkpoint_epoch_number))
    resume_model = load_model(checkpoint_epoch_path)
    return resume_model, checkpoint_epoch_number
    
`
#######################################################################################################    
def main(unused_args):
    # Create the Estimator
    estimator = tf.estimator.Estimator(cnn_model_fn)
    print ("Training...")
    estimator.train(input_fn=lambda: input_fn(train_filepath,
                                                batch_size=10,
                                                num_epochs=10,
                                                shuffle=True),steps = 10)
    print ("Validating...")
    evaluation = estimator.evaluate(input_fn=lambda: input_fn(val_filepath,
                                                            batch_size=3,
                                                            num_epochs=3,
                                                            shuffle=False))
    print(evaluation)

    # Export trained model as SavedModel.
    receiver_fn = serving_input_receiver_fn
    estimator.export_savedmodel(export_path, receiver_fn)

#######################################################################################################
#Runs the program with an optional "main" function and "argv" list. DEPRECATED IN 2.0
if __name__ == "__main__":
    tf.app.run(main)
