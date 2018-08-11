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

let lastTweet = 'This is my tweet'

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
    f: 'tweet'
  }
  
  getModel()
    .then(model => 
      T.get('search/tweets', params, (err, { statuses }) => {
        if (err) return console.log(err)
        const [xs, ys] = prepareBatch(statuses)

        fs.writeFile(`network/data/${Date.now()}.json`, JSON.stringify(statuses))

        lastTweet = statuses.slice(-1)[0].text

        model.fit(xs, ys, {
          epochs: 10,
          batchSize: statuses.length,
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

setInterval(sendTweet, +process.env.TWEET_INTERVAL)
trainModel()