const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

const model = tf.sequential({
  layers: [
    tf.layers.dense({ units: 10, inputShape: [10, 1] }),
    tf.layers.lstm({ units: 512, returnSequences: true, recurrentActivation: 'softplus' }),
    tf.layers.dense({ units: 10 })
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
  epochs: 3000,
  batchSize,
  callbacks: {
    onEpochEnd: (epoch, log) => {
      console.log(`Epoch ${epoch}: loss = ${log.loss}`)
    },
    onTrainEnd: () => {
      console.log('End of training')
      model.save('file://build/mymodel')
    }
  }
})