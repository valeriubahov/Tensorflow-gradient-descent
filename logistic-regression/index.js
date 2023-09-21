const tf = require("@tensorflow/tfjs");
const loadCSV = require("../utils/load-csv");
const LogisticRegression = require("./logistic-regression");
const { createChart } = require("../utils/chartGenerator");

const { features, labels, testFeatures, testLabels } = loadCSV("../data/cars.csv", {
  dataColumns: ["horsepower", "displacement", "weight"],
  labelColumns: ["passedemissions"],
  shuffle: true,
  splitTest: 50,
  converters: {
    passedemissions: (value) => {
      return value.toUpperCase() === "TRUE" ? 1 : 0;
    },
  },
});

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
  decisionBoundary: 0.5,
});

regression.train();

const numberOfIterations = Array.from(Array(regression.options.iterations).keys());
createChart(numberOfIterations, regression.costEntropyHistory, "Cross Entropy");

const accuracy = regression.test(testFeatures, testLabels);
console.log(`The Model has a prediction accuracy of ${accuracy * 100}%`);
