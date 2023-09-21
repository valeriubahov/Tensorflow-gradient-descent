const { ChartJSNodeCanvas } = require("chartjs-node-canvas");
const fs = require("fs");

const width = 800; //px
const height = 800; //px
const borderColor = "red";
const backgroundColor = "white";

const chartCallback = (ChartJS) => {
  ChartJS.defaults.responsive = true;
  ChartJS.defaults.maintainAspectRatio = false;
};

const createChart = async (xValues, yValues, chartName) => {
  const configuration = {
    type: "line", // define the chart type
    data: {
      // X line
      labels: xValues,
      datasets: [
        // Y line
        {
          label: chartName,
          data: yValues,
          borderWidth: 1,
          borderColor: borderColor,
          pointStyle: "circle",
          pointRadius: 2,
          pointHoverRadius: 10,
        },
      ],
    },
    options: {},
    plugins: [
      // define the background color of the chart
      {
        id: "background-colour",
        beforeDraw: (chart) => {
          const ctx = chart.ctx;
          ctx.save();
          ctx.fillStyle = backgroundColor;
          ctx.fillRect(0, 0, width, height);
          ctx.restore();
        },
      },
      {
        legend: {
          position: "top",
        },
        title: {
          display: true,
          text: chartName,
        },
      },
    ],
  };

  const chartJSNodeCanvas = new ChartJSNodeCanvas({ width, height, chartCallback });
  const buffer = chartJSNodeCanvas.renderToBufferSync(configuration);
  fs.writeFileSync(`../charts/${chartName}.png`, buffer, "base64");
};

module.exports = { createChart };
