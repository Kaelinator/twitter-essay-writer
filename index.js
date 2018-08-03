const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

const inputs = tf.input({shape: [3, 10]})

const rnn = tf.layers.rnn({
  cell: [
    tf.layers.lstmCell({units: 5}),
    tf.layers.lstmCell({units: 12}),
    tf.layers.lstmCell({units: 1}),
  ],
  returnSequences: true
})

const outputs = rnn.apply(inputs)

const model = tf.model({ inputs, outputs })

model.predict(tf.randomNormal([5, 3, 10])).print()



// model.compile({
//   optimizer: 'sgd',
//   loss: 'meanSquaredError'
// })

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