require('dotenv').config()

const Twit = require('twit')

const getModel = require('./src/network')
const { toInput, toTweet, prepareBatch } = require('./src/data')
const fs = require('fs')

const T = new Twit({
  consumer_key: process.env.CONSUMER_KEY,
  consumer_secret: process.env.CONSUMER_SECRET,
  access_token: process.env.ACCESS_TOKEN,
  access_token_secret: process.env.ACCESS_TOKEN_SECRET
})

// let lastTweet = 'This is my tweet'
let lastTweet = 'zabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz'

const sendTweet = () => {
  console.log('Tweet time!')
  
  getModel()
  .then(model => {
    console.log('input: ', JSON.stringify(lastTweet))
    model.predict(toInput(lastTweet))
      .data()
      .then(toTweet)
      .then(status => {
        console.log('Tweet:', JSON.stringify(status))
          T.post('statuses/update', { status }, (err, data) => {
            if (err) console.log('Error tweeting: ', err)
          })
      })
  })
}

const predictTweet = () => {
  console.log('Predict time!')
  
  getModel()
  .then(model => {
    console.log('input: ', JSON.stringify(lastTweet))
    model.predict(toInput(lastTweet))
      .data()
      .then(toTweet)
      .then(status => {
        console.log('Tweet:', JSON.stringify(status))
      })
  })
}
  
const trainModel = () => {
    
  console.log('training...')

  const params = {
    lang: 'en',
    q: 'essay OR college OR write OR story',
    count: 100,
    result_type: 'recent',
    tweet_mode: 'extended',
    f: 'tweet'
  }
  
  getModel()
    .then(model => 
      T.get('search/tweets', params, (err, { statuses }) => {
        if (err) return console.log(err)

        // const tweets = statuses.map(status => status.full_text)
        // const avgLength = tweets.reduce((sum, t) => t.length + sum, 0) / tweets.length
        // const purgedTweets = tweets.filter(t => t.length > avgLength)
        const purgedTweets = Array(128)
          .fill('abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz')
          .map((x, i) => i % 2 === 0 ? 'z' + x : x)
        const [xs, ys] = prepareBatch(purgedTweets)

        fs.writeFile(`network/data/${Date.now()}.json`, JSON.stringify(purgedTweets), (err) => err && console.log('error writing file:', err))

        // lastTweet = purgedTweets.slice(-1)[0]

        model.fit(xs, ys, {
          epochs: 36,
          batchSize: purgedTweets.length,
          callbacks: {
            onEpochEnd: (epoch, log) => {
              console.log(`Epoch ${epoch}: loss = ${log.loss}`)
            },
            onTrainEnd: () => {
              console.log('End of training')
              model.save(process.env.MODEL_PATH)
              predictTweet()
              trainModel() // do it again
            }
          }
        })
      })
    )
}

// setInterval(sendTweet, +process.env.TWEET_INTERVAL)
trainModel()