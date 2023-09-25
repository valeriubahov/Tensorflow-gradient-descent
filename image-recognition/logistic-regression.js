const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LogisticRegression {
  constructor(features, labels, options) {
    // Create tensors from arrays for the features and labels
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);

    // here I'll store the  cross entropy value in order to adjust the learning rate automatically
    // how incorrect we are over time
    this.costEntropyHistory = [];

    // record how our guesses of B are changing over time
    this.bHistory = [];

    // set default value to VERY important property
    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000, // how many time to run our Linear Descent logarithm before exit
        decisionBoundary: 0.5,
      },
      options
    );

    // initial tensor containing b and m
    // shape of weights will be same as number of columns of features and labels
    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
  }

  gradientDescent(features, labels) {
    // EQUATION ==> (Features * ((Features * Weights) - Labels)) / n

    // Features => Tensor of feature data
    // Labels => Tensor of label data
    // n => Number of observations
    // Weights => Tensor containing M and B
    // Features * Weights => Tensor Moltiplication (Matrix moltiplication)

    // matMul => matrix moltiplication (number of columns of tensor1 must be equal to number of rows of tensor 2)
    // (Features * Weights)
    // Tensorflow has a softmax function to return the probability if the value to be 0 or 1
    const currentGuesses = features.matMul(this.weights).softmax();

    // ((Features * Weights) - Labels)
    const differences = currentGuesses.sub(labels);

    // transpose => make rows become columns and columns become rows
    // (Features * ((Features * Weights) - Labels))) / n
    const slopes = features.transpose().matMul(differences).div(features.shape[0]);

    // update the values of b and m
    // m,b =  m/b - (mslope/bslope * learningRate)
    return this.weights.sub(slopes.mul(this.options.learningRate));
  }

  // train our model
  train() {
    // How many batches of data we have
    const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);

    for (let i = 0; i < this.options.iterations; i++) {
      // calculate the gradient descent using batch of data
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const { batchSize } = this.options;

        this.weights = tf.tidy(() => {
          const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]);
          const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

          return this.gradientDescent(featureSlice, labelSlice);
        });
      }

      // save B history for analytics purpose
      this.bHistory.push(this.weights.get(0, 0));

      // after every gradient descent calculation record the mean square error
      this.recordCost();

      // update the learning rate to use for the next training loop
      this.updateLearningRate();
    }
  }

  // observations = [[horsepower, weight, displacement]]
  predict(observations) {
    // standardize the observations and return the predictions
    // 0 = FAIL
    // 1 = PASS
    return this.processFeatures(observations).matMul(this.weights).softmax().argMax(1);
  }

  // run some tests with our trained model
  test(testFeatures, testLabels) {
    // predictions will always return a probability to return a 1 rather than a 0
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels).argMax(1);

    // count how many incorrect predictions we got
    const incorrect = predictions.notEqual(testLabels).sum().get();

    // return the percentage of correct answers (accuracy)
    return (predictions.shape[0] - incorrect) / predictions.shape[0];
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
    return tf.ones([features.shape[0], 1]).concat(features, 1);
  }

  standardize(features) {
    // calculate the mean and the variance to use to standardize our features
    const { mean, variance } = tf.moments(features, 0);
    // variance = 0 cause division by zero causing NaN values.
    // SOLUTION: cast 0 to bool => FALSE
    // use logicalNot from tensorflow to transform false into TRUE and cast again to number
    const filler = variance.cast("bool").logicalNot().cast("float32");
    // Save the mean and variance values inside our class to use to standardize our Test features
    // Test features MUST use the same mean and variance that we used to standardize our features
    this.mean = mean;
    this.variance = variance.add(filler);

    // return the standardized value
    return features.sub(mean).div(this.variance.pow(0.5));
  }

  recordCost() {
    const cost = tf.tidy(() => {
      const guesses = this.features.matMul(this.weights).softmax();

      const termOne = this.labels.transpose().matMul(guesses.add(1e-7).log());
      const termTwo = this.labels
        .mul(-1)
        .add(1)
        .transpose()
        .matMul(guesses.mul(-1).add(1).add(1e-7).log());
      // log(0) is -infinity that cause problems finding the accuracy
      //by adding 1e-7 we ensure the we never do log(0)

      return termOne.add(termTwo).div(this.features.shape[0]).mul(-1).get(0, 0);
    });

    this.costEntropyHistory.push(cost);
  }

  updateLearningRate() {
    // update the learning rate by checking the last 2 cross entropy values
    if (this.costEntropyHistory.length < 2) {
      return;
    }

    // get last 2 entropy calculations from history
    const lastCost = this.costEntropyHistory.at(-1);
    const secondLastCost = this.costEntropyHistory.at(-2);

    if (lastCost > secondLastCost) {
      // entropy increased from last calculation then divide learning rate by 2
      this.options.learningRate /= 2;
    } else {
      // entropy decresed then we speed up the learning rate by 5%
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;
