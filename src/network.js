const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

module.exports = () => tf.loadModel(process.env.MODEL_PATH + 'model.json')
  .catch((err) => {

    console.log('Error loading model, generating new model:',
      process.MODEL_PATH, JSON.stringify(err))

    const model = tf.sequential({
      layers: [
        tf.layers.dense({ units: +process.env.TWEET_LENGTH, inputShape: [+process.env.TWEET_LENGTH, 1] }),
        tf.layers.lstm({
          units: 512,
          returnSequences: true,
          recurrentActivation: 'softplus'
        }),
        tf.layers.dense({ units: +process.env.RANGE })
      ]
    })
    
    model.compile({
      optimizer: 'sgd',
      loss: 'meanSquaredError'
    })

    return model
  })