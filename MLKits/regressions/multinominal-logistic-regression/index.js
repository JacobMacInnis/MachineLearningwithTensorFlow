require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');


function loadData() {
    let mnistData = mnist.training(0, 60000);
    const features = mnistData.images.values.map(image => _.flatMap(image));


    const encodedLabels = mnistData.labels.values.map(label => {
        const row = new Array(10).fill(0);
        row[label] = 1;
        return row;

    });
    return { features, labels: encodedLabels };
}
const { features, labels } = loadData();
const regression = new LogisticRegression(features, labels, {
    learningRate: 1,
    iterations: 40,
    batchSize: 500,
});

regression.train();
const testMnistData = mnist.testing(0, 10000);

const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));
const testEncodedLabels = testMnistData.labels.values.map(label => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;

})

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log('Accuracy is: ', accuracy);

plot({
    x: regression.costHistory.reverse()
})



// const loadCSV = require('../load-csv');
// const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
//     dataColumns: [
//         'horsepower',
//         'displacement',
//         'weight',
//     ],
//     labelColumns: ['mpg'],
//     shuffle: true,
//     splitTest: 50,
//     converters: {
//         mpg: (value) => {
//             const mpg = parseFloat(value);

//             if (mpg < 15) {
//                 return [1, 0, 0];
//             } else if (mpg < 30) {
//                 return [0, 1, 0];
//             } else {
//                 return [0, 0, 1];
//             }
//         }
//     },
// });

// const regression = new LogisticRegression(features, _.flatMap(labels), {
//     learningRate: 0.5,
//     iterations: 100,
//     batchSize: 10,

// });

// // regression.weights.print();


// regression.train();
// // regression.predict([
// //     [150, 200, 2.223]
// // ]).print();
// // // regression.predict([
// // //     [130, 307, 1.75]
// // // ]).print()

// console.log(regression.test(testFeatures, _.flatMap(testLabels)))

// // plot({
// //     x: regression.costHistory.reverse()
// // })