const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LinearRegression {
  constructor(features, labels, options) {
    this.features = features;
    this.labels = labels;
    this.options = Object.assign(
      // set default value to VERY important property
      {
        learningRate: 0.1,
        iterations: 1000, // how many time to run our Linear Descent logarithm before exit
      },
      options
    );

    // initial guesses of m and b
    this.m = 0;
    this.b = 0;
  }

  gradientDescent() {
    // 1 - find the MSE with respect to B
    // 2 - find the MSE with respect to M

    // EQUIATION:
    // 2/numOfRecords * (sum((m * feature + b) - label))

    const currentGuessesForMPG = this.features.map((row) => {
      // row[0] = horse power
      return this.m * row[0] + this.b;
    });

    // calculate the MSE of B
    const bSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, index) => {
          return guess - this.labels[index][0];
        })
      ) *
        2) /
      this.features.length;

    // calculate the MSE of M
    const mSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, index) => {
          return -1 * this.features[index][0] * (this.labels[index][0] - guess);
        })
      ) *
        2) /
      this.features.length;

    // update the MSE slops for m and b
    this.m = this.m - mSlope * this.options.learningRate;
    this.b = this.b - bSlope * this.options.learningRate;
  }

  // train our model
  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }
}

module.exports = LinearRegression;
