# -*- coding: utf-8 -*-
"""
@author: Sam Cheung

Python programme to train, evaluate and predict Author based on obfuscated text using a character CNN model.
"""

import os
import json
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import optimizers, callbacks
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D
from keras.layers import GlobalMaxPooling1D, Add, Flatten, Convolution1D, Concatenate
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("model_name", None, "Name of model")

flags.DEFINE_string(
    "data_dir", None,
    "The directory that contains the .txt files for model training and "
    "prediction.")

flags.DEFINE_string(
    "output_dir", None,
    "The directory to store the trained model and predictions.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run model evaluation.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the testing dataset.")

flags.DEFINE_integer(
    "max_len", 455,
    "The maximum length of character sequence for the input sentence. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("batch_size", 128, "The batch_size for training and prediction.")

flags.DEFINE_integer("num_epochs", 100, "The maximum number of training epochs.")

flags.DEFINE_integer("early_stopping_patience", 10,
                     "The number of epochs with no improvement after which "
                     "training will be stopped.")

flags.DEFINE_integer("reduce_lr_patience", 5,
                     "The number of epochs with no improvement after which learning "
                     "rate will be reduced.")

flags.DEFINE_float("dropout_rate", 0.5, 
                   "The dropout rate in convolutional layer and feed-forward "
                   "neural network layer.")

flags.DEFINE_float("reduce_lr_factor", 0.5, 
                   "The factor applied to reduce learning rate.")

flags.DEFINE_float("proportion_for_validation", 0.2,
                   "The proportion of training data reserved for validation.")

flags.DEFINE_float("learning_rate", 0.001, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_convolution_layers", 8, "Number of convolutional layer "
                     "in the model.")

flags.DEFINE_integer("num_filters", 64, 
                     "Number of filters in each convolutional layer.")

flags.DEFINE_integer("strides", 1, "Number of strides in convolution.")

flags.DEFINE_integer("num_classes", 12, "Number of classes in the classification task.")


def sentence2index(sentence):
  """a function to map a sentence of characters to number indexes"""
  charList = list(sentence)
  indexList = [ord(char)-96 for char in charList]
  return indexList


class characterCNN(object):
  
  """a class for character Convolutional Neural Network"""
  
  def __init__(self, max_len, embed_weight, 
               num_convolution_layers, num_filters, kernel_size,
              strides, dropout_rate, num_classes, pretrained_model=None):  
    self.max_len = max_len
    self.embed_weight = embed_weight
    self.num_convolution_layers = num_convolution_layers 
    self.num_filters = num_filters 
    self.kernel_size = kernel_size 
    self.strides = strides
    self.dropout_rate = dropout_rate
    self.num_classes = num_classes
    self.model = pretrained_model
    
  def build_model(self):
    """
    Build the character CNN model
    
    """
    print("Building character CNN model...")
    
    # Input layer, output dimension: (batch_size, max_len)
    sequence_input = Input(shape=(self.max_len,), dtype='int32', name="input_layer")
    # Embedding layer, output dimension: (batch_size, max_len, 26)
    # the embed_weight has a dimension of (27,26), which results from the row concatenation of zero row vector (for
    # padding index) and an one-hot embedding matrix (for the 26 English characters)
    embedded_sequences = Embedding(27, 26, weights=[self.embed_weight], 
                                   input_length=self.max_len, 
                                   trainable=False,
                                  name="one_hot_embedding_layer")(sequence_input)

    max_pool_list = []
    
    for i in range(self.num_convolution_layers):
      # Convolutional layer, output dimension for each layer: (batch_size, max_len - kernel_size + 1, num_filters)
      # Layers of varying kernel size captures the semantics of character combinations of different lengths
      conv = Convolution1D(filters=self.num_filters[i], 
                           kernel_size=self.kernel_size[i], 
                           strides=self.strides[i], 
                           name="conv_"+str(i+1))(embedded_sequences)
      # Dropout
      dropout = Dropout(self.dropout_rate, name="dropout_"+str(i+1))(conv)
      # Maxpooling layer, output dimension for each layer: (batch_size, num_filters)
      # This layer serves to further extract important semantic features for prediction.
      max_pool = GlobalMaxPooling1D(name="max_pooling_1D_" + str(i+1))(dropout)
      max_pool_list.append(max_pool)
      
    # Concatenation layer, output dimension: (batch_size, num_convolution_layers*num_filters)
    concat = Concatenate(axis=1, name="concatenated_max_pool")(max_pool_list)
    # Feed-forward neural network layer, output dimension: (batch_size, 1024)
    output = Dense(1024, activation="relu", name="feed_forward")(concat)
    output = Dropout(self.dropout_rate, name="feed_forward_dropout")(output)
    # output layer, output dimension: (batch_size, num_classes)
    output = Dense(self.num_classes, activation='softmax', name="output_layer")(output)

    model = Model(inputs=sequence_input, outputs=output)
    self.model = model
    
    print("Character CNN model built: ")
    self.model.summary()
                   
  
  def train(self, train_inputs, train_labels, 
            valid_inputs, valid_labels,
            learning_rate, 
            early_stopping_patience, reduce_lr_patience, reduce_lr_factor, 
            num_epochs, batch_size):
    
    """Train the character CNN"""
    # compile the model, using categorical crossentropy as loss function,
    # Adam as optimizer
    self.model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])

    # Early Stopping, should the validation loss exceeds that of previous epoch
    # for more than 10 times, the training of the network will be stopped.
    early_stopping_callback = EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=early_stopping_patience,
                                            verbose=1, 
                                            mode='auto',
                                            restore_best_weights=True)
    
    # Should the validation loss exceeds that of previous epoch for more than 5 
    # times, the learning rate applied to the network will be halved. This is 
    # to avoid the network from over-shooting when approaching the minimum of 
    # loss. The initial learning rate is 0.001.
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss',
                                           factor=reduce_lr_factor,
                                           patience=reduce_lr_patience,
                                           verbose=1)
    
    print("Begin model training...")
    self.model.fit(train_inputs,
              train_labels,
              epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(valid_inputs, valid_labels),
              verbose=1,
              callbacks=[early_stopping_callback, 
                         reduce_lr_callback])
  
          
  def evaluate_trained_model(self, test_inputs, test_labels, test_labels_one_hot,
                            test_batch_size):
    
    """evaluate trained model"""      
    
    loss, accuracy = self.model.evaluate(test_inputs, test_labels_one_hot)
    predicted_probits = self.model.predict(test_inputs, 
                                    batch_size=test_batch_size, verbose=1)
    predicted_labels = np.argmax(predicted_probits, axis=1)

    print("Prediction accuracy: {0:.3f}\nLoss: {1:.5f}".format(accuracy, loss))
    print(classification_report(test_labels, predicted_labels)) # recall, precision and f-1 score

  
  def predict_new_batch(self, test_inputs, test_batch_size, output_dir):
    
    """Perform batch prediction"""      
    
    predicted_probits = self.model.predict(test_inputs, 
                                           batch_size=test_batch_size, 
                                           verbose=1)
    predicted_labels = np.argmax(predicted_probits, axis=1)
    
    print('Prediction completed!')
    print('Saving prediction to {}'.format(output_dir,))
    with open(os.path.join(output_dir, "prediction.txt"), "w") as f:
      for label in list(predicted_labels):
        f.write("{}\n".format(label))
  
  def save(self, save_path):
    """save trained model"""
    self.model.save(save_path)
	
	
def main(_):
  """main function"""
  
  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
  
  np.random.seed(12345) # for reproducible results
  
  if FLAGS.do_train:

    print("Importing training data...")
    with open(os.path.join(FLAGS.data_dir, "xtrain_obfuscated.txt")) as f:
      x_raw = f.read().splitlines()

    with open(os.path.join(FLAGS.data_dir, "ytrain.txt")) as f:
      y_raw = f.read().splitlines()

    x = list(map(sentence2index, x_raw))
    x = pad_sequences(x, maxlen=FLAGS.max_len)

    y = [int(char) for char in y_raw]
    y = to_categorical(y)
    
    #split data into training and validation set 
    print('{0:.0%} of data serves as validation set...'.format(FLAGS.proportion_for_validation))
    valid_idx = np.random.choice(range(len(x)), size=int(FLAGS.proportion_for_validation*len(x)))
    train_idx = np.setdiff1d(range(len(x)), valid_idx)

    xtrain = x[train_idx]
    ytrain = y[train_idx]
    ytrain_raw = np.array(y_raw)[train_idx]

    xvalid = x[valid_idx]
    yvalid = y[valid_idx]
    yvalid_raw = np.array(y_raw)[valid_idx]

    embed_weight = np.concatenate((np.zeros([1,26]), np.identity(26)))
    
    classifier = characterCNN(max_len=FLAGS.max_len, 
                              embed_weight=embed_weight,
                              num_convolution_layers=FLAGS.num_convolution_layers, 
                              num_filters=[FLAGS.num_filters]*FLAGS.num_convolution_layers, 
                              kernel_size=list(range(3,3+FLAGS.num_convolution_layers)),
                              strides=[FLAGS.strides]*FLAGS.num_convolution_layers, 
                              dropout_rate=FLAGS.dropout_rate, 
                              num_classes=FLAGS.num_classes)
    
    # save the model configuration file
    model_config = {'model_name':FLAGS.model_name,
                    'max_len':FLAGS.max_len,
                    'batch_size':FLAGS.batch_size,
                    'num_epochs':FLAGS.num_epochs,
                    'early_stopping_patience':FLAGS.early_stopping_patience,
                    'reduce_lr_patience':FLAGS.reduce_lr_patience,
                    'reduce_lr_factor':FLAGS.reduce_lr_factor,
                    'dropout_rate':FLAGS.dropout_rate,
                    'proportion_for_validation':FLAGS.proportion_for_validation,
                    'learning_rate':FLAGS.learning_rate,
                    'num_convolution_layers':FLAGS.num_convolution_layers,
                    'num_filters':FLAGS.num_filters,
                    'strides':FLAGS.strides,
                    'num_classes':FLAGS.num_classes}

    with open(os.path.join(FLAGS.output_dir, FLAGS.model_name+'_config.json'), 'w') as fw:
        json.dump(model_config, fw) 
    
    
    # build model
    classifier.build_model()
    # train model
    classifier.train(train_inputs=xtrain, train_labels=ytrain,
                     valid_inputs=xvalid, valid_labels=yvalid,
                     learning_rate=FLAGS.learning_rate,
                     early_stopping_patience=FLAGS.early_stopping_patience, 
                     reduce_lr_patience=FLAGS.reduce_lr_patience, 
                     reduce_lr_factor=FLAGS.reduce_lr_factor,
                     num_epochs=FLAGS.num_epochs, 
                     batch_size=FLAGS.batch_size)
    
    print("Training completed, saving model to {}".format(os.path.join(FLAGS.output_dir, 
                                                                       FLAGS.model_name+".hdf5")))
    classifier.save(os.path.join(FLAGS.output_dir, FLAGS.model_name+".hdf5"))
    
    # evaluate model performance
    print("Evaluating model performance on training dataset...")
    classifier.evaluate_trained_model(test_inputs=xtrain, 
                                      test_labels=[int(char) for char in ytrain_raw],
                                      test_labels_one_hot=ytrain,
                                      test_batch_size=FLAGS.batch_size)
    print("Evaluating model performance on validation dataset...")
    classifier.evaluate_trained_model(test_inputs=xvalid, 
                                      test_labels=[int(char) for char in yvalid_raw],
                                      test_labels_one_hot=yvalid,
                                      test_batch_size=FLAGS.batch_size) 
    
    
  if FLAGS.do_eval:
    
    # load model
    print("Loading trained model...")
    model_path = os.path.join(FLAGS.output_dir, FLAGS.model_name+".hdf5")
    trained_model = load_model(model_path)
    embed_weight = np.concatenate((np.zeros([1,26]), np.identity(26)))
    
    classifier = characterCNN(max_len=FLAGS.max_len, 
                              embed_weight=embed_weight,
                              num_convolution_layers=FLAGS.num_convolution_layers, 
                              num_filters=[FLAGS.num_filters]*FLAGS.num_convolution_layers, 
                              kernel_size=list(range(3,3+FLAGS.num_convolution_layers)),
                              strides=[FLAGS.strides]*FLAGS.num_convolution_layers, 
                              dropout_rate=FLAGS.dropout_rate, 
                              num_classes=FLAGS.num_classes,
                              pretrained_model=trained_model)      

    # import train files
    with open(os.path.join(FLAGS.data_dir, "xtrain_obfuscated.txt")) as f:
      x_raw = f.read().splitlines()

    with open(os.path.join(FLAGS.data_dir, "ytrain.txt")) as f:
      y_raw = f.read().splitlines()

    x = list(map(sentence2index, x_raw))
    x = pad_sequences(x, maxlen=FLAGS.max_len)

    y = [int(char) for char in y_raw]
    y = to_categorical(y)

    #split data into training and validation set 
    valid_idx = np.random.choice(range(len(x)), size=int(FLAGS.proportion_for_validation*len(x)))
    train_idx = np.setdiff1d(range(len(x)), valid_idx)

    xtrain = x[train_idx]
    ytrain = y[train_idx]
    ytrain_raw = np.array(y_raw)[train_idx]

    xvalid = x[valid_idx]
    yvalid = y[valid_idx]
    yvalid_raw = np.array(y_raw)[valid_idx]

    # evaluate model performance
    print("Evaluating model performance on training dataset...")
    classifier.evaluate_trained_model(test_inputs=xtrain, 
                                      test_labels=[int(char) for char in ytrain_raw],
                                      test_labels_one_hot=ytrain,
                                      test_batch_size=FLAGS.batch_size)
    print("Evaluating model performance on validation dataset...")
    classifier.evaluate_trained_model(test_inputs=xvalid, 
                                      test_labels=[int(char) for char in yvalid_raw],
                                      test_labels_one_hot=yvalid,
                                      test_batch_size=FLAGS.batch_size)
    
  if FLAGS.do_predict:
    
    print("Performing prediction on testing data set...")
    model_path = os.path.join(FLAGS.output_dir, FLAGS.model_name+".hdf5")
    trained_model = load_model(model_path)
    embed_weight = np.concatenate((np.zeros([1,26]), np.identity(26)))
    
    classifier = characterCNN(max_len=FLAGS.max_len, 
                              embed_weight=embed_weight,
                              num_convolution_layers=FLAGS.num_convolution_layers, 
                              num_filters=[FLAGS.num_filters]*FLAGS.num_convolution_layers, 
                              kernel_size=list(range(3,3+FLAGS.num_convolution_layers)),
                              strides=[FLAGS.strides]*FLAGS.num_convolution_layers, 
                              dropout_rate=FLAGS.dropout_rate, 
                              num_classes=FLAGS.num_classes,
                              pretrained_model=trained_model)     

    with open(os.path.join(FLAGS.data_dir, "xtest_obfuscated.txt")) as f:
      xtest_raw = f.read().splitlines()

    xtest = list(map(sentence2index, xtest_raw))
    xtest = pad_sequences(xtest, maxlen=FLAGS.max_len)

    classifier.predict_new_batch(test_inputs=xtest, 
                                 test_batch_size=FLAGS.batch_size, 
                                 output_dir=FLAGS.output_dir)    
								 
								 
if __name__ == "__main__":
  # required data field before running training/evaluation/prediction  
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("model_name")
  tf.app.run()
