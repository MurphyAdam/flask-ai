### Flask-AI
Simple Flask app exposing endpoint for sentiment analysis of text.
The model is trained on [IMDB reviews](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) with Tensorflow/Keras
Dataset of 20k reviews for training, and 5k for validation.

An input is condidered positive if it has a score of > 0.65 

### Install
You need Python and PIP installed.
Create an env var and activate it, then install the requirements.txt

```sh
virtualenv venv
source venv/bin/activate   
pip install -r requirements.txt 
```
### Run
```sh
python wsgi.py
```
or
```sh
gunicorn wsgi:app --bind 127.0.0.1:8001
```

### API
Once you start the server with either commands above, head to `127.0.0.1:8001` in your browser and test the app.
The `/api/predict` returns an object in the structure:

```json
{
  "sentiment": "positive",
  "text": "This is very nice",
  "weight": 0.711793839931488
}
```

### Deployment
This is deployed to Heroku with the trained model under `sentiment_model` directory.

### Live URL
`https://flask-ai-5fdcbdbfb317.herokuapp.com/`

### Issues faced
> Needed to experiment with the parameters of the model to get a not an okay result.
> Longer texts are not classified well.
> Heroku slug limit of 500mb on deployment reached bacuse of tensorflow's size. Used tensorflow CPU for deployment.

### Examples:

```
[0.50000393]: Text: very bad, Sentiment: negative
[0.7262488 ]: Text: i love it, Sentiment: positive
[0.5       ]: Text: waste of time, Sentiment: negative
[0.5       ]: Text: awful movie, Sentiment: negative
[0.7310585 ]: Text: amazing movie, Sentiment: positive
[0.5499662 ]: Text: I wish I never gave it a try, Sentiment: negative
[0.5000005 ]: Text: bad, Sentiment: negative
[0.73024505]: Text: good, Sentiment: positive
[0.7245549 ]: Text: not my cup of tea, Sentiment: positive
[0.50032616]: Text: could have been better, Sentiment: negative
[0.5004509 ]: Text: I want my money back, Sentiment: negative
[0.7310581 ]: Text: i enjoyed it, Sentiment: positive
[0.73105764]: Text: great time, Sentiment: positive
[0.50021017]: Text: I find it boring, Sentiment: negative
[0.73105836]: Text: the settings were amazing, the story as well, Sentiment: positive
[0.50218534]: Text: the actor was not that good, Sentiment: negative
[0.7041709 ]: Text: nicest movie of the year, Sentiment: positive
[0.5       ]: Text: dont waste your time with this, Sentiment: negative
[0.52289206]: Text: I just hate it, Sentiment: negative
[0.5000406 ]: Text: it's okay, Sentiment: negative
[0.7277739 ]: Text: it's fine, Sentiment: positive
[0.50000393]: Text: very bad, Sentiment: negative
[0.67680097]: Text: mild, not good, not bad, Sentiment: positive
```