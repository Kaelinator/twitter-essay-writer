require('dotenv').config()

const Twit = require('twit')

const getModel = require('./src/network')
const { toInput, toTweet, prepareBatch } = require('./src/data')

const T = new Twit({
  consumer_key: process.env.CONSUMER_KEY,
  consumer_secret: process.env.CONSUMER_SECRET,
  access_token: process.env.ACCESS_TOKEN,
  access_token_secret: process.env.ACCESS_TOKEN_SECRET
})

const sendTweet = () => {
  console.log('Tweet time!')
  
  getModel()
  .then(model => {
    
    model.predict(toInput('This is my tweet'))
    .data()
    .then(toTweet)
    .then(console.log)
    // .then(status => {
    //   T.post('statuses/update', { status }, (err, data) => {
    //     if (err) console.log('Error: ', err)
    //     if (data) console.log('Data: ', data)
    //   })
    // })
  })
}
  
const trainModel = () => {
    
  console.log('training...')

  const params = {
    language: 'en',
    q: 'essay OR college OR write',
    count: 1000
  }
  
  getModel()
  .then(model => 
    T.get('search/tweets', params, (err, { statuses }) => {
      if (err) return console.log(err)

      const [xs, ys] = prepareBatch(statuses)
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
            trainModel() // do it again
          }
        }
      })
    })
  )
}

setInterval(sendTweet, +process.env.TWEET_INTERVAL)
trainModel()