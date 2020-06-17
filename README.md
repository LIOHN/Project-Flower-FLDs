# Project-Flower-FLDs
Solo Masters project "COMP592 Fashion Landmark Detection with a Convolutional Neural Network" 

In this project, I took on the task of training a machine learning model to predict key points on pictures of garments with the use of CNNs. The challenge was to find a robust and performant architecture that would beat a benchmark model called DFA (deep fashion alignment). 

This model was trained using multiple Nvidia GTX 1080-Ti GPUs with preprocessed training data from the DeepFashion Dataset (http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html).

The training data contained 123k images (that were preprocessed with functions from the datasetToTFR.py file) with a training-validation-testing split of 70-15-15.
The model definition is contained in the model.py file. 
Running landmark.py sets off the training process, and picks up work from last checkpoint (if there is one).

Results were average/below-average in comparison with the benchmark model (one that also used the DeepFashion Dataset). The benchmark model was much more powerful than mine thanks to its cascading structure and coarse-to-fine approach. In my results, some landmarks are perfectly predicted, while others are not very close to where they should be.

One challenge I was faced with in this project was the training loop. My code features an automatic training loop, making use of the TFRecord processing functions. With a manually coded training loop (example in one Stanford University CS20 Deep Learning Tutorial https://github.com/chiphuyen/stanford-tensorflow-tutorials), recording and visualising results and debugging would be a much smoother process with the use of TensorFlow's Tensorboard.
