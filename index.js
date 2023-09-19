import("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const LinearRegression = require("./linear-regression");

let { features, labels, testFeatures, testLabels } = loadCSV("./data/cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["mpg"],
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 100,
});

regression.train();

// miles per gallons, we are not sure about the precision yet
const r2 = regression.test(testFeatures, testLabels);
// if r2 is negative that means that our previsions are higher than the actual values
// - infinite < r2 < 1
console.log("R2 is", r2);
