const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

const getPrediction = predictions => {
  const chunk = 10
  const temparray = []
  for (let i = 0, j = predictions.length; i < j; i += chunk)
      temparray.push(predictions.slice(i, i + chunk))
  return temparray.map(arr => arr.indexOf(Math.max(...arr)))
}

tf.loadModel('file://build/mymodel/model.json')
  .then(model => {

    model
      .predict(tf.tensor3d([[ [1], [1], [1], [2], [1], [3], [1], [4], [1], [5] ]]))
      .data()
      .then(getPrediction)
      .then(console.log)

    model
      .predict(tf.tensor3d([[ [0], [1], [2], [3], [4], [5], [6], [7], [8], [9] ]]))
      .data()
      .then(getPrediction)
      .then(console.log)
  })
