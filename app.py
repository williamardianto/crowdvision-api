from PIL import Image
from facenet_pytorch import MTCNN
from flask import Flask, request, jsonify
import torch
import os
import urllib.request
import pymongo
from bson.binary import Binary
import pickle as cPickle

mongo_client = pymongo.MongoClient('mongodb://mongo:27017')
hunter_db = mongo_client['hunter_db']
hunter_collection = hunter_db['hunter_collection']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device='cpu')
cosine_dist = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

application = Flask(__name__)

try:
    model = torch.jit.load("/tmp/InceptionResnetV1-jit.pth", map_location=torch.device('cpu'))
    model.eval()
except ValueError:
    application.logger.info('value error, create folder and downloading model')
    urllib.request.urlretrieve('https://www.dropbox.com/s/f0jr1qagnna7gd4/InceptionResnetV1-jit.pth',
                               '/tmp/InceptionResnetV1-jit.pth')
    model = torch.jit.load("/tmp/InceptionResnetV1-jit.pth", map_location=torch.device('cpu'))
    model.eval()

@application.route("/test", methods=['POST'])
def test():
    hunter_collection.insert_one({'test':'test'})
    return 'Hello, dffdfd!'

def get_embedding(img):
    with torch.no_grad():
        img_cropped = mtcnn(img)
        embedding = model(img_cropped.unsqueeze(0))
    return embedding


def to_pickle_binary(embedding):
    return Binary(cPickle.dumps(embedding, protocol=2))


def to_tensor(pickle_binary):
    return(cPickle.loads(pickle_binary))


def compute_distance(all_data, input_vector):
    max = 0.8
    id = None
    for x in all_data:
            dist=cosine_dist(to_tensor(x['embedding']), input_vector).item()
            if dist > max:
                max = dist
                id = x['_id']

    return id

@application.route("/")
def index():
    return 'Hello, World!'


@application.route("/search", methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file)
    embedding = get_embedding(img)

    all_data = hunter_collection.find()

    max_id = compute_distance(all_data, embedding)

    if max_id is not None:
        instance = hunter_collection.find({'_id': max_id})
        result = list(instance)[0]
        output_dict = {
            'name': result['name'],
            'age': result['age'],
            'desc': result['desc']
        }
        return jsonify(output_dict)
    else:
        output_dict = {
            'message': 'not match'
        }
        return jsonify(output_dict)

@application.route("/upload", methods=['POST'])
def upload():
    name = request.form.get('name')
    age = request.form.get('age')
    desc = request.form.get('desc')

    file = request.files['file']
    img = Image.open(file)
    embedding = get_embedding(img)

    if embedding is not None:
        dict = {
            'name': name,
            'age': age,
            'desc': desc,
            'embedding': to_pickle_binary(embedding)
        }
        hunter_collection.insert_one(dict)
    else:
        return "no face detected"

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000)
