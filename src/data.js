const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

const toData = tweet => Array(+process.env.TWEET_LENGTH)
  .fill(0)
  .map((_, i) => tweet.charCodeAt(i)) // get code
  .map(c => c < +process.env.RANGE && c || 0) // no emojis

const toInput = tweet => tf.tensor1d(Int32Array.from(toData(tweet))).reshape([1, +process.env.TWEET_LENGTH, 1])

const toTweet = array => {
  const temparray = []

  for (let i = 0, j = array.length; i < j; i += +process.env.RANGE)
      temparray.push(array.slice(i, i + +process.env.RANGE))

  return String.fromCharCode(...temparray
    .map(arr => arr.indexOf(Math.max(...arr))))
}

const flatten = (a, t) => a.concat(t)

const prepareBatch = (tweets) => {

  const data = tweets
    .map(tweet => tweet.text)
    .map(toData)
  
  const xs = tf.tensor1d(
    Int32Array.from(data.reduce(flatten, []))
  , 'int32').reshape([tweets.length, +process.env.TWEET_LENGTH, 1])
  
  const shiftedData = data.map(tweet => [...tweet.slice(1), 0])

  const ys = tf.oneHot(
    tf.tensor1d(Int32Array.from(shiftedData.reduce(flatten, [])), 'int32')
  , +process.env.RANGE).reshape([tweets.length, +process.env.TWEET_LENGTH, +process.env.RANGE]).toFloat()

  return [xs, ys]
}
  
module.exports = {
  toInput,
  toTweet,
  prepareBatch
}