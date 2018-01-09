# A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification



## 1. Background

### CNN Architecture

begin with a tokenized sentence -> a sentence matrix

sentence matrix의 row는 word vector들.

word vector들은 pretrained 사용



word vectors dimension: d

length of a given sentence: s

*sentence matrix: s x d*



treat sentence matrix as an ‘image’

perform convolution on it via linear filters

filters with widths equal to the dimensionality of the word vectors (d)

height of the filter: the region size of the filter.



denote the sentence matrix by A

A[i:j]: the sub-matrix of A from row i to row j.



w: weight matrix

b: bias term

The output sequence of the convolution operator: $o_i = w \cdot{A[i:j+h-1]}$

feature map: $c_i = f(o_i + b)$ 	$c \in R^{s-h+1}$

A pooling function is thus applied to each feature map to induce a fixed-length vector.

A common strategy is 1-max pooling

the outputs generated from each filter map can be concatenated into a fixed-length, ‘top-level’ feature vector

then fed through a softmax function to generate the final classification.



## 2. Baseline Models



![Imgur](https://i.imgur.com/s7aXR1N.png)

**CNN architecture for sentence classification**

region sizes -> gram

feature map -> convolution 결과

필터 개수만큼의 feature map을 만들고, Max-pooling 과정을 거쳐 클래스 개수만큼의 스코어를 출력



We kept only the most frequent 30k n-grams for all datasets

tuned hyperparameters via nested cross-fold validation

incorporating word2vec embeddings into feature vectors



### configuration

a linear kernel SVM exploiting uni- and bi-gram features

used averaged word vectors (from Google word2vec or GloVe)



![Imgur](https://i.imgur.com/L7FZiC9.png)



## references

A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional
Neural Networks for Sentence Classification(2016) - Ye Zhang, Byron C. Wallace

http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/

https://ratsgo.github.io/natural%20language%20processing/2017/03/19/CNN/