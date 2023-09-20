import("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./utils/load-csv");
const { createChart } = require("./utils/chartGenerator");

const LinearRegression = require("./linear-regression");
let { features, labels, testFeatures, testLabels } = loadCSV("./data/cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["mpg"],
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 5,
  batchSize: 10, // define the batch size for batch gradient descent
});

regression.train();

// miles per gallons, we are not sure about the precision yet
const r2 = regression.test(testFeatures, testLabels);

// if r2 is negative that means that our previsions are higher than the actual values
// - infinite < r2 < 1
console.log(`Guesses accuracy is ${Math.round(r2 * 100, 2)}%`);

// create a linear chart to view the Plot between MSE and B
createChart(regression.bHistory, regression.mseHistory, "Mean Squared Error");

const numberOfIterations = Array.from(Array(regression.options.iterations).keys());
createChart(numberOfIterations, regression.mseHistory, "Iterations #");
