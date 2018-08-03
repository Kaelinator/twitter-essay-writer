const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

const cells = [
  tf.layers.lstmCell({units: 4}),
  tf.layers.lstmCell({units: 8})
]

const model = tf.sequential({ 
  layers: [
    tf.layers.dense({ units: 1, activation: 'sigmoid', inputShape: [2] }),
    tf.layers.rnn({cell: cells, returnSequences: true }),
    // tf.layers.dense({ units: 1, activation: 'sigmoid' }),
    tf.layers.dense({ units: 1, activation: 'sigmoid' })
    // tf.layers.lstm({ units: 100, activation: 'sigmoid', returnSequences: true }),
  ]
})

model.compile({
  optimizer: 'sgd',
  loss: 'meanSquaredError'
})

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