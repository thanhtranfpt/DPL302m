import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__, template_folder='./')
CORS(app)

# Loading our model and tokenizer into our backend. Now our backend can use these to make prediction.
model = load_model('model.h5', compile=False)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# When you go to localhost:5000 using a web browswer, it will render index.html's content
@app.route('/')
def index():
    return render_template('index.html')

# The Frontend will make API call to this localhost:5000/predict?text=TheMovieIsTerrible, and backend will return "The review is negative"
@app.route('/predict', methods=['GET'])
def predict():
    text = request.args.get('text', '')
    print(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=120)
    prediction = model.predict(padded)
    if prediction[0][0] >= 0.5:
         return "The review is positive. " + text
    else:
        return "The review is negative."