# Recognition of handwritten digits using a neural network
<div align="center">
<i>Christopher Shaffer, Shuyang Jiang, and Zehui Chen</i>
</div>
<br/>
 
## Summary
In this project, we focused on the application of feedforward neural networks with backpropagation
to character recognition. The model of feedforward neural networks was studied in detail along with the
backpropagation algorithm, which is important in evaluating how the network changes in response to small
changes. We also studied the algorithms for training which include stochastic-gradient backpropagation
algorithm and its two variants. The softmax training changes the activation functions in the output layer
into a normalization step. The cross entropy training modifies the risk function with a cross-entropy
formulation. The two variants were considered in order to counter the slowdown of learning thus improve the
performance.

The three algorithms were implemented and tested with the [Chars74k dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/). Four groups
of data from the dataset were used for testing, including "natural scenes" uppercase characters, handwritten
characters, handwritten digits and computer generated fonts. The performance of certain algorithms on
certain data was measured in terms of 1 - R<sub>test;</sub> which is the rate of correct recognition by the trained
network, and displayed on a heatmap spanned by its two parameters, the regularization factor &rho; and the
step size &mu;. We also compared the performance of different activation functions, softplus and sigmoid, for
the stochastic-gradient training algorithm.

As a result, we found that softplus outperforms sigmoid because it prevents saturation. Among the three algorithms, the softmax algorithm has better performance than the
other two. Among the four datasets, the computer generated font dataset has the best performance because
the neural network can be trained sufficiently on its large number of sample. When the neural network is
trained on computer generated uppercase font data with the softmax algorithm, the network correctly recognized 87% of the testing characters.

## Data files and MATLAB scripts
The datasets and MATLAB code used in this project can be found [here](https://github.com/chrisshaffer/matlab-handwritten-digits/tree/main/MATLAB%20Scripts%20and%20Data%20Files).

## Report
An in-depth which explains the theory, algorithm performance comparisons, and hyperparameter optimization can be found [here](https://github.com/chrisshaffer/matlab-handwritten-digits/blob/main/Final%20Report.pdf).
