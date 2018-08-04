const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

const rnn = tf.layers.rnn({
  cell: [
    tf.layers.lstmCell({units: 50})
  ],
  returnSequences: true,
  inputShape: [10, 1] // [timesteps, feature]
})

const model = tf.sequential({
  layers: [
    rnn,
    tf.layers.dense({ units: 10 })
  ]
})


model.compile({
  optimizer: 'sgd',
  loss: 'meanSquaredError'
})

/* [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 1, 1, ... ] */
const data = Int32Array.from(Array(1000)
  .fill(0)
  .reduce((arr, x, i) => arr.concat((i).toString().split('')), []), x => +x)

const ddata = Array.from(data.slice(1)).reduce((a, x) => { 
  const arr = Array(10).fill(0)
  arr[x] = 1
  return a.concat(arr)
}, [])

const xs = tf.tensor1d(data).reshape([289, 10, 1])
const ys = tf.tensor1d([...ddata, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape([289, 10, 10])

model.fit(xs, ys, {
  epochs: 1,
  batchSize: 289,
  callbacks: {
    onEpochEnd: (epoch, log) => {
      console.log(`Epoch ${epoch}: loss = ${log.loss}`)
    },
    onTrainEnd: () => {
      console.log('End of training')
      model.predict(tf.tensor3d([[ [1], [1], [1], [2], [1], [3], [1], [4], [1], [5] ]]))
        .print()
    }
  }
})