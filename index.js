const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

const rnn = tf.layers.rnn({
  cell: [
    tf.layers.lstmCell({units: 5}),
    tf.layers.lstmCell({units: 12}),
    tf.layers.lstmCell({units: 1}),
  ],
  returnSequences: true
})

const model = tf.sequential({
  layers: [
    tf.layers.dense({ units: 50, inputShape: [3, 10]}),
    rnn
  ]
})


model.compile({
  optimizer: 'sgd',
  loss: 'meanSquaredError'
})

model.predict(tf.randomNormal([5, 3, 10])).print()

// const xs = tf.randomNormal([100, 2])
// const ys = tf.randomNormal([100, 1])

// xs.print()
// ys.print()

// model.fit(xs, ys, {
//   epochs: 100,
//   callbacks: {
//     onEpochEnd: (epoch, log) => {
//       console.log(`Epoch ${epoch}: loss = ${log.loss}`)
//     },
//     onTrainEnd: (x, y, z) => {
//       console.log('x', x, 'y', y, 'z', z)
//     }
//   }
// })