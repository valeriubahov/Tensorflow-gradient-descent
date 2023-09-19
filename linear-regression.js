const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LinearRegression {
  constructor(features, labels, options) {
    // Create tensors from arrays for the features and labels
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);

    // set default value to VERY important property
    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000, // how many time to run our Linear Descent logarithm before exit
      },
      options
    );

    // initial tensor containing b and m
    this.weights = tf.zeros([this.features.shape[1], 1]);
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
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

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

  processFeatures(features) {
    features = tf.tensor(features);

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    // MUST APPLY ONLY AFTER STANDARDIZATION
    // create a column of ones in order to make possible the tensor moltiplication (matrix moltiplication)
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  standardize(features) {
    // calculate the mean and the variance to use to standardize our features
    const { mean, variance } = tf.moments(features, 0);

    // Save the mean and variance values inside our class to use to standardize our Test features
    // Test features MUST use the same mean and variance that we used to standardize our features
    this.mean = mean;
    this.variance = variance;

    // return the standardized value
    return features.sub(mean).div(variance.pow(0.5));
  }
}

module.exports = LinearRegression;
