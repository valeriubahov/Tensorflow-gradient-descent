const tf = require("@tensorflow/tfjs");
const loadCSV = require("../utils/load-csv");
const LogisticRegression = require("./logistic-regression");
const { createChart } = require("../utils/chartGenerator");
const _ = require("lodash");

// Predict whether a car is low, medium or high efficiency based on horsepower, displacement and weight

const { features, labels, testFeatures, testLabels } = loadCSV("../data/cars.csv", {
  dataColumns: ["horsepower", "displacement", "weight"],
  labelColumns: ["mpg"],
  shuffle: true,
  splitTest: 50,
  converters: {
    mpg: (value) => {
      const mpg = parseFloat(value);
      // [low, medium, hight]
      if (mpg < 15) {
        return [1, 0, 0];
      } else if (mpg < 30) {
        return [0, 1, 0];
      } else {
        return [0, 0, 1];
      }
    },
  },
});

const regression = new LogisticRegression(features, _.flatMap(labels), {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
  decisionBoundary: 0.5,
});

regression.train();

const accuracy = regression.test(testFeatures, _.flatMap(testLabels));

console.log(`The Model has a prediction accuracy of ${accuracy * 100}%`);
