const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LinearRegression {
  constructor(features, labels, options) {
    // Create tensors from arrays for the features and labels
    this.features = tf.tensor(features);
    this.labels = tf.tensor(labels);

    // create a column of ones in order to make possible the tensor moltiplication (matrix moltiplication)
    this.features = tf.ones([this.features.shape[0], 1]).concat(this.features, 1);

    // set default value to VERY important property
    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000, // how many time to run our Linear Descent logarithm before exit
      },
      options
    );

    // initial tensor containing b and m
    this.weights = tf.zeros([2, 1]);
  }

  gradientDescent() {
    // EQUATION ==> (Features * ((Features * Weights) - Labels)) / n

    // Features => Tensor of feature data
    // Labels => Tensor of label data
    // n => Number of observations
    // Weights => Tensor containing M and B
    // Features * Weights => Tensor Moltiplication (Matrix moltiplication)

    // matMul => matrix moltiplication (number of columns of tensor1 must be equal to number of rows of tensor 2)
    // (Features * Weights)
    const currentGuesses = this.features.matMul(this.weights);

    // ((Features * Weights) - Labels)
    const differences = currentGuesses.sub(this.labels);

    // transpose => make rows become columns and columns become rows
    // (Features * ((Features * Weights) - Labels))) / n
    const slopes = this.features.transpose().matMul(differences).div(this.features.shape[0]);

    // update the values of b and m
    // m,b =  m/b - (mslope/bslope * learningRate)
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  // train our model
  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }

  test(testFeatures, testLabels) {
    // convert arrays to tensors
    testFeatures = tf.tensor(testFeatures);
    testLabels = tf.tensor(testLabels);

    // add a column of ones to our test features to enable matrix moltiplication
    testFeatures = tf.ones([testFeatures.shape[0], 1]).concat(testFeatures, 1);

    // predict the values using b and m calculated before in the training phase
    const predictions = testFeatures.matMul(this.weights);

    // From here we got the predictions but we don't know if they are accurate
    // To check the accuracy of our predictions we are going to use the 'Coefficient of Determination'

    // Coefficient of Determination => 1 - SSres / SStot
    // SSres => SUM of (Labels - Predictions)^2
    // SStot => SUM of (Labels - Average of Labels)^2

    // SSres
    const ss_res = testLabels.sub(predictions).pow(2).sum().get();

    //SStot
    const ss_tot = testLabels.sub(testLabels.mean()).pow(2).sum().get();

    return 1 - ss_res / ss_tot; // coefficient of determination
  }
}

module.exports = LinearRegression;
