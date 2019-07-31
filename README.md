
# obfuscated-multiclassification
The task requires training a deep learning model that classifies a given line of text as belonging to one of the following 12 novels:

* alice_in_wonderland
* dracula
* dubliners
* great_expectations
* hard_times
* huckleberry_finn
* les_miserable
* moby_dick
* oliver_twist
* peter_pan
* talw_of_two_cities
* tom_sawyer

The data has been obfuscated while the patterns are preserved. The obfuscated data are sequences of lower case English letters concatenated without any space character.

## Methodology
A character-based Convolutional Neural Network (CNN) is trained for this task. Given that the English characters are concatenated without any space character, it is intuitive to treat the combination of adjacent characters of varying length as **n-grams**. Hence convolution layers with different kernel size in the convolutional filters are used to extract the semantics of these **n-grams** independently. 

In the experiment, 20% of the original training data serve as validation data to keep track of the model training performance. The validation set also serves for early stopping criteria to be mentiond below.

## Model Architecture
### Input Layer
Given one input obfuscated text sequence, each English character is mapped to a unique integer index (1 to 26 in this case). The maximum length of the text sequence in the dataset is about 455. Hence sequences shorter than this length will be padded with **0** at the beginning of the sequences.

### Embedding Layer
The embedding layer is initialized with an embedding weight of dimension (27,26). It is the concatenation of a 26-dimensional zero row vector and a one-hot matrix (or identity matrix). The zero row vector corresponds to the padding index 0 and each of the 26 one hot vector refer to each character index. The embedding layer is set to be **non-trainable**. Thus the embedding layer serves to map each integer index to its corresponding one-hot vector (except "0" which is a vector of zero). For one input sequence, its output (or **Embedded Sequence**), from the Embedding Layer has the dimension (455,26).

### Convolutional Layer
**8** convoluational layers are adopted in the model. Each convolutional layer has **64** convolution filters while 8 kernel sizes are adopted (range from 3 to 10, increment by 1) in each convolutional layer. The 8 convolutions are applied to the **Embedded Sequence** **simultanously and independently**, followed by ReLU activaion. 8*64 = 512 activation maps are obtained after the convolution.

Fewer convolutional layers have been experimented but it turns out 8 layers perform the best in terms of prediction accuracy. More convolutional layers do not necessary outperform 8 layers and hence 8 layers are used in consideration of model parsimony. 

### Maxpooling Layer
The activation maps obtained from each convolution layer are maxpooled and concatenated into a single vector respectively, which further extract useful information in the "**ngrams**" of different sizes for Novel prediction. Hence there are eight 64-dimensional vectors after maxpooling.

Average pooling have been taken into consideration as well (keeping other hyperparameters constant), yet it underperforms Maxpooling significantly. It might have diluted the salient features for prediction from each activation maps.

### Concatenation Layer
The eight 64-dimensional vectors from the Maxpooling layers are further concantented row-wise to form a 512-dimensional vector. The vector is the input to the feed-forward neural network layer.

### Feed-forward Neural Network Layer
Only one feed-forward neural network layer with ReLU activation is used in the model after several trails for varying network layer. More layers do not necessary lead to higher prediction accuracy but increase model size. Therefore a more parsimonious model is chosen. 

Fewer number of nodes have been experimented but a node size of 1024 yields the best result.

### Output Layer
The final layer uses the Softmax function to output the 12 prediction probabilities.

## Measures to avoid overfitting
The following handles are added to the model to prevent the model from overfitted during training:
* **Dropout**
Dropout is applied in convolutional layer and feed-forward neural network layer with a rate of 0.5 to force different neural network weights to learn all features instead of inclining to learn one/some of them.
* **Early Stopping**
Should the validation loss exceeds that of previous epoch for more than 10 times, the training of the network will be stopped.
* **Reduce learning rate on plateau**
Should the validation loss exceeds that of previous epoch for more than 5 times, the learning rate applied to the network will be halved. This is to avoid the network from over-shooting when approaching the minimum of loss. The initial learning rate is 0.001.

## Getting Started
### Python3 Library Prerequisites
`json==2.0.9`  
`numpy==1.16.4`  
`keras==2.2.4`  
`tensorflow==1.13.1`  
`sklearn==0.21.2`  

## Running the tests

Model Training
```sh
$ python obfuscatedclassification.py --data_dir=[directory of the training data] \
                                     --output_dir=[directory to save model/prediction] \
                                     --model_name=[Name of the model] \
                                     --do_train=True
```
Model Evaluation
```sh
$ python obfuscatedclassification.py --data_dir=[directory of the data] \
                                     --output_dir=[directory to save model/prediction] \
                                     --model_name=[Name of the model] \
                                     --do_eval=True
```
Running the model on testing dataset
```sh
$ python obfuscatedclassification.py --data_dir=[directory of the testing data] \
                                     --output_dir=[directory to save model/prediction] \
                                     --model_name=[Name of the model] \
                                     --do_predict=True
```

## Results
**Training Data**  
`Prediction accuracy: 0.990`  
`Weighted Precision: 0.99`  
`Weighted Recall: 0.99`  
`Weighted f1-score: 0.99`  
`Loss: 0.09308`  

|               | precision |   recall | f1-score |support|
|---------------|-----------|----------|----------|-------|                
|    Novel 0    |    1.00   |   1.00   |   1.00   |  445  |
|    Novel 1    |    0.99   |   1.00   |   0.99   | 2873  |
|    Novel 2    |    0.99   |   0.99   |   0.99   | 1210  |
|    Novel 3    |    0.99   |   0.98   |   0.99   | 3288  |
|    Novel 4    |    0.99   |   0.98   |   0.99   | 1905  |
|    Novel 5    |    1.00   |   0.99   |   0.99   | 1876  |
|    Novel 6    |    0.99   |   1.00   |   0.99   | 3416  |
|    Novel 7    |    1.00   |   1.00   |   1.00   | 4174  |
|    Novel 8    |    0.99   |   0.99   |   0.99   | 2985  |
|    Novel 9    |    0.99   |   0.96   |   0.98   |  805  |
|    Novel10    |    0.98   |   0.98   |   0.98   | 2482  |
|    Novel11    |    0.96   |   0.99   |   0.98   | 1158  |
|    accuracy   |           |          |   0.99   | 26617 |
|   macro avg   |    0.99   |   0.99   |   0.99   | 26617 |
|weighted avg   |    0.99   |   0.99   |   0.99   | 26617 |


**Validation Data**  
`Prediction accuracy: 0.859`  
`Weighted Precision: 0.86`  
`Weighted Recall: 0.85`  
`Weighted f1-score: 0.85`  
`Loss: 0.43943`  

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| Novel 0      | 0.86      | 0.86   | 0.86     | 112     |
| Novel 1      | 0.84      | 0.86   | 0.85     | 641     |
| Novel 2      | 0.85      | 0.76   | 0.80     | 291     |
| Novel 3      | 0.82      | 0.82   | 0.82     | 807     |
| Novel 4      | 0.91      | 0.87   | 0.89     | 483     |
| Novel 5      | 0.98      | 0.90   | 0.94     | 445     |
| Novel 6      | 0.86      | 0.88   | 0.87     | 895     |
| Novel 7      | 0.88      | 0.92   | 0.90     | 1013    |
| Novel 8      | 0.92      | 0.88   | 0.90     | 726     |
| Novel 9      | 0.89      | 0.84   | 0.86     | 189     |
| Novel 10     | 0.76      | 0.77   | 0.77     | 623     |
| Novel 11     | 0.73      | 0.85   | 0.79     | 277     |
| accuracy     |           |        | 0.86     | 6502    |
| macro avg    | 0.86      | 0.85   | 0.85     | 6502    |
| weighted avg | 0.86      | 0.86   | 0.86     | 6502    |

Based on the performance of the model on the validation data, **the expected accuracy, weighted precision, recall and f-1 score on testing data are about 0.85**.

## Author

* **Sam Cheung**
