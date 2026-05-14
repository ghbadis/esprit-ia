from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

model = pickle.load(open('model_categorie.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    titre = data.get('titre', '')
    description = data.get('description', '')
    text = titre + ' ' + description
    X = vectorizer.transform([text])
    categorie = model.predict(X)[0]
    return jsonify({'categorie': categorie})

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'API IA Esprit fonctionne!'})

if __name__ == '__main__':
    app.run(debug=True)