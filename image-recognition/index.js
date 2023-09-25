const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./logistic-regression");
const { createChart } = require("../utils/chartGenerator");
const _ = require("lodash");
const mnist = require("mnist-data");

// garbage collector will remove mnistData automatically decreasing memory usage
function loadData() {
  const mnistData = mnist.training(0, 60000);

  const features = mnistData.images.values.map((image) => _.flatMap(image));

  const encodedLabels = encodeLabels(mnistData.labels.values);

  return { features, labels: encodedLabels };
}

function loadTestData() {
  // preparing test data
  const testMnistData = mnist.testing(0, 10000);
  const testFeatures = testMnistData.images.values.map((image) => _.flatMap(image));
  const testEncodedLabels = encodeLabels(testMnistData.labels.values);

  return { testFeatures, testEncodedLabels };
}

const { features, labels } = loadData();

const regression = new LogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 40,
  batchSize: 100,
});

regression.train();

const { testFeatures, testEncodedLabels } = loadTestData();
const accuracy = regression.test(testFeatures, testEncodedLabels);

console.log("Accuracy is", accuracy);
const numberOfIterations = Array.from(Array(regression.options.iterations).keys());

createChart(numberOfIterations, regression.costEntropyHistory, "image_costHystory");

function encodeLabels(labels) {
  return labels.map((label) => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
  });
}
