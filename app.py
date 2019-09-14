from PIL import Image
from facenet_pytorch import MTCNN
from flask import Flask, request, abort, jsonify, send_from_directory
import torch
import os
import urllib.request
import pymongo
from bson.binary import Binary
import pickle as cPickle
import string
import random

import json
from flask import Response
from werkzeug.utils import secure_filename
from datetime import date

from bson.objectid import ObjectId

mongo_client = pymongo.MongoClient('mongodb://mongo:27017')
hunter_db = mongo_client['hunter_db']
hunter_collection = hunter_db['hunter_collection']
matching_collection = hunter_db['matching_collection']

mtcnn = MTCNN(keep_all=False, device='cpu')
cosine_dist = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

application = Flask(__name__)

try:
    model = torch.jit.load("./model/InceptionResnetV1-jit.pth", map_location=torch.device('cpu'))
    model.eval()
except ValueError:
    application.logger.info('value error, create folder and downloading model')
    os.mkdir("./model")
    urllib.request.urlretrieve('https://www.dropbox.com/s/f0jr1qagnna7gd4/InceptionResnetV1-jit.pth',
                               '/tmp/InceptionResnetV1-jit.pth')
    model = torch.jit.load("/tmp/InceptionResnetV1-jit.pth", map_location=torch.device('cpu'))
    model.eval()


UPLOAD_DIRECTORY = os.path.join(os.getcwd(),'images')
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

def resize_image(img):
    width, height = img.size
    img = img.resize((width // 3, height // 3), Image.ANTIALIAS)
    return img

def get_embedding(img):
    embedding = None
    with torch.no_grad():
        img_cropped = mtcnn(img)
        if img_cropped is not None:
            embedding = model(img_cropped.unsqueeze(0))
    return embedding


def to_pickle_binary(embedding):
    return Binary(cPickle.dumps(embedding, protocol=2))


def to_tensor(pickle_binary):
    return(cPickle.loads(pickle_binary))


def compute_distance(all_data, input_vector):
    max = 0
    id = None
    dist_list = []
    for x in all_data:
            dist=cosine_dist(to_tensor(x['embedding']), input_vector).item()
            dist_list.append(dist)
            if dist > max:
                max = dist
                id = x['_id']

    return id, max, dist_list

@application.route("/")
def index():
    return 'Hello, World!'

# @application.route("/test", methods=['POST'])
# def test():
#     hunter_collection.insert_one({'test':'test'})
#     return 'Hello, World!'


@application.route("/search", methods=['POST'])
def predict():
    longitude = request.form.get('longitude')
    latitude = request.form.get('latitude')
    file = request.files['file']
    suspect = save_file(file)
    img = Image.open(file)
    img = resize_image(img)
    embedding = get_embedding(img)

    if embedding is None:
        return jsonify({'message': 'no face detected'})
    else:
        all_data = hunter_collection.find()
        max_id, dist, dist_list = compute_distance(all_data, embedding)

        instance = hunter_collection.find({'_id': max_id})
        result = list(instance)[0]
        output_dict = {
            'nama': result['nama'],
            'umur': result['umur'],
            'desc': result['desc'],
            'jantina': result['jantina'],
            'bangsa': result['bangsa'],
            'filename': result['filename'],
            'confidence': dist,
            'type': result['type']
            # 'distances': str(dist_list)
        }

        if dist > 0.7:

            match = {
                'nama': result['nama'],
                'umur': result['umur'],
                'desc': result['desc'],
                'jantina': result['jantina'],
                'bangsa': result['bangsa'],
                'filename': result['filename'],
                'type': result['type'],
                'confidence': dist,

                'suspect': suspect,
                'longitude': longitude,
                'latitude': latitude
            }
            matching_collection.insert_one(match)

        return jsonify(output_dict)



@application.route("/upload", methods=['POST'])
def upload():
    nama = request.form.get('nama')
    umur = request.form.get('umur')
    desc = request.form.get('desc')
    type = request.form.get('type')
    jantina = request.form.get('jantina')
    bangsa = request.form.get('bangsa')

    file = request.files['file']
    filename = save_file(file)
    img = Image.open(file)
    img = resize_image(img)
    embedding = get_embedding(img)

    if embedding is not None:
        dict = {
            'nama': nama,
            'umur': umur,
            'desc': desc,
            'type': type,
            'jantina': jantina,
            'bangsa': bangsa,
            'embedding': to_pickle_binary(embedding),
            'createdDate': str(date.today()),
            'filename': filename
        }
        hunter_collection.insert_one(dict)

        return jsonify({'message':'uploaded'})
    else:
        return jsonify({'message':'no face detected'})

@application.route("/criminals", methods=['GET'])
def get_criminals():
    criminals = hunter_collection.find({'type':'criminal'})
    criminals_list = list(criminals)
    output_list = []
    for c in criminals_list:
        d = {}
        d['id'] = str(c['_id'])
        d['nama'] = c['nama']
        d['umur'] = c['umur']
        d['type'] = c['type']
        d['desc'] = c['desc']
        d['jantina'] = c['jantina']
        d['bangsa'] = c['bangsa']
        d['filename'] = c['filename']
        d['createdDate'] = c['createdDate']
        output_list.append(d)

    return Response(json.dumps(output_list),  mimetype='application/json')

@application.route("/missings", methods=['GET'])
def get_missings():
    criminals = hunter_collection.find({'type':'missing'})
    criminals_list = list(criminals)
    output_list = []
    for c in criminals_list:
        d = {}
        d['id'] = str(c['_id'])
        d['nama'] = c['nama']
        d['umur'] = c['umur']
        d['type'] = c['type']
        d['desc'] = c['desc']
        d['jantina'] = c['jantina']
        d['bangsa'] = c['bangsa']
        d['filename'] = c['filename']
        d['createdDate'] = c['createdDate']
        output_list.append(d)

    return Response(json.dumps(output_list),  mimetype='application/json')

@application.route("/delete", methods=['DELETE'])
def delete():
    id = request.form.get('id')
    hunter_collection.delete_one({"_id": ObjectId(id)})
    return jsonify({'message':'deleted'})

@application.route("/match", methods=['GET'])
def get_match():
    suspects = matching_collection.find()
    suspects_list = list(suspects)
    output_list = []
    for c in suspects_list:
        d = {}
        d['id'] = str(c['_id'])
        d['nama'] = c['nama']
        d['umur'] = c['umur']
        d['type'] = c['type']
        d['desc'] = c['desc']
        d['jantina'] = c['jantina']
        d['bangsa'] = c['bangsa']
        d['filename'] = c['filename']
        d['suspect'] = c['suspect']
        d['longitude'] = c['longitude']
        d['latitude'] = c['latitude']
        d['confidence'] = c['confidence']
        output_list.append(d)

    return Response(json.dumps(output_list),  mimetype='application/json')


#########################################3
def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def save_file(file):
    filename = randomString(20) +'.jpg'
    # filename = file.filename
    path = os.path.join(UPLOAD_DIRECTORY, filename)
    file.save(path)

    filename = 'file/'+filename
    return filename

@application.route("/file/<path:filename>")
def get_file(filename):
    """Download a file."""
    return send_from_directory(UPLOAD_DIRECTORY, filename, as_attachment=True)


if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000)
