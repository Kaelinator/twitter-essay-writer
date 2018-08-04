const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

const model = tf.sequential({
  layers: [
    tf.layers.dense({ units: 16, activation: 'linear', inputShape: [10, 1] }),
    tf.layers.lstm({ units: 128, returnSequences: true, recurrentActivation: 'softplus' }),
    tf.layers.dense({ units: 10, activation: 'sigmoid' })
  ]
})


model.compile({
  optimizer: 'sgd',
  loss: 'meanSquaredError'
})

/* [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 1, 1, ... ] */
const data = Int32Array.from(Array(100)
  .fill(0)
  .reduce((arr, x, i) => arr.concat((i).toString().split('')), []), x => +x)

const batchSize = data.length / 10

const xs = tf.tensor1d(data, 'int32').reshape([batchSize, 10, 1])
const ys = tf.oneHot(tf.tensor1d([...data.slice(1), 0], 'int32'), 10).reshape([batchSize, 10, 10]).toFloat()

model.fit(xs, ys, {
  epochs: 100,
  batchSize,
  callbacks: {
    onEpochEnd: (epoch, log) => {
      console.log(`Epoch ${epoch}: loss = ${log.loss}`)
    },
    onTrainEnd: () => {
      console.log('End of training')
      const x = [[ [1], [1], [1], [2], [1], [3], [1], [4], [1], [5] ]]
      model.save('file://mymodel')
      model
        .predict(tf.tensor3d(x))
        .data()
        .then(predictions => {
          const chunk = 10
          const temparray = []
          for (let i = 0, j = predictions.length; i < j; i += chunk)
              temparray.push(predictions.slice(i, i + chunk))
          return temparray
        }, [[]])
        .then(predictions => predictions.map(arr => arr.indexOf(Math.max(...arr))))
        .then(console.log)
    }
  }
})