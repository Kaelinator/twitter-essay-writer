const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')
const Promise = require('bluebird')

const compConfig = {
  optimizer: 'sgd',
  loss: 'categoricalCrossentropy'
}
let globalModel = null

module.exports = () => globalModel 
  ? Promise.resolve(globalModel) 
  : tf.loadModel(process.env.MODEL_PATH + 'model.json')
    .then(model => {

      model.compile(compConfig)

      globalModel = model
      return model
    })
    .catch(() => {

      console.log('Error loading model, generating new model:', process.env.MODEL_PATH)

      const model = tf.sequential({
        layers: [
          tf.layers.lstm({
            units: +process.env.LSTM_UNITS,
            returnSequences: true,
            inputShape: [+process.env.TWEET_LENGTH, 1]
          }),
          tf.layers.dense({ units: +process.env.RANGE })
        ]
      })

      model.compile(compConfig)

      globalModel = model
      return model
    })
