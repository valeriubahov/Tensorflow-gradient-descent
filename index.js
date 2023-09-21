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
  iterations: 3,
  batchSize: 10, // define the batch size for batch gradient descent
});

regression.train();

// TEST ARE USED ONLY DURING DEVELOPMENT, NOT NECESSARY IF WE ARE MAKING PREDICTIONS
// miles per gallons, we are not sure about the precision yet
const r2 = regression.test(testFeatures, testLabels);
// if r2 is negative that means that our previsions are higher than the actual values
// - infinite < r2 < 1

// create a linear chart to view the Plot between MSE and B
createChart(regression.bHistory, regression.mseHistory, "Mean Squared Error_BATCH");

// create a linear chart that indicate the Plot between MSE and the number of iterations
const numberOfIterations = Array.from(Array(regression.options.iterations).keys());
createChart(numberOfIterations, regression.mseHistory, "Iterations #_BATCH");

// PUT SOME PREDICTIONS DATA -> MUST FOLLOW FEATURES ORDER
// horsepower, weight, displacement
const dataToPredict = [
  [120, 2, 380],
  [135, 2.5, 420],
];

// Predict the value
const predictions = regression.predict(dataToPredict);

// Prin the results
const numberOfPredictions = predictions.shape[0];
console.log("Predicted Miles x Gallon:");
console.log("", "");

for (let i = 0; i < numberOfPredictions; i++) {
  console.log(
    `Car ${i + 1}
    Horsepower: ${dataToPredict[(0, i)][0]} hp
    Weight: ${dataToPredict[(0, i)][1]} Tons
    Displacement: ${dataToPredict[(0, i)][2]} cui
    Can run ${predictions.get(i, 0).toFixed(2)} MPG`
  );
  console.log("", "");
}

console.log(`The predictions accuracy is: ${Math.round(r2 * 100, 2)}%`);
console.log("", "");
