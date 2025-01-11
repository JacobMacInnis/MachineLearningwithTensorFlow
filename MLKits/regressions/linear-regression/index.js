require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 3,
    batchSize: 10
});

regression.train();

// console.log('Updated M is: ', regression.weights.arraySync()[1][0]);
// console.log('Updated B is: ', regression.weights.arraySync()[0][0]);
console.log(regression.test(testFeatures, testLabels));

plot({
    x: regression.mseHistory.reverse(),
    xLabel: 'Iteration #',
    yLabel: 'Mean Squared Error'
})

regression.predict([
    [79, 1.3125, 120],
    [84, 1.1475, 135]
]).print();