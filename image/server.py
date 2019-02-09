#!/usr/bin/env python

from flask import Flask, Response, request, abort
from functools import wraps
import ssl
import json
import cv2
import numpy as np
import jsonpickle
import base64
from werkzeug import secure_filename
import tempfile
import match
import uuid

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('server.crt', 'server.key')

app = Flask(__name__)

def require_appkey(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        input_key = ""
        if request.args.get('key'):
            input_key = request.args.get('key')
        if request.headers.get('x-api-key'):
            input_key = request.headers.get('x-api-key')

        try:
            apikey = open('api.key', 'r')
            key=apikey.readline().replace('\n', '')
            
            while key:
                if input_key == key:
                    return view_function(*args, **kwargs)
                key = apikey.readline().replace('\n', '')
        finally:
            apikey.close()

        abort(401)
    return decorated_function

def name_to_file(name):
    try:
        json_data = json.loads(open('images.json').read())
        return json_data[name]
    except:
        return None

def save_image(name, filename):
    try:
        json_data = json.loads(open('images.json').read())            
    except:
        json_data = {}

    json_data[name] = filename
    with open(json_file, 'w') as json_file:
        json.dump(json_decoded, json_file)    
    
    

@app.route('/api/v1/compare', methods=['POST'])
@require_appkey
def compare():
    json_str = request.json
    json_out = json.loads(json_str)
    
    if json_out['image'] is None or json_out['name'] is None:
        response = {'message': 'missing image or name'}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")
    
    image = json_out['image']
    ground_truth = name_to_file(json_out['name'])
    if ground_truth is None:
        response = {'message': 'invalid name'}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")

    data = base64.b64decode(image)
    image_file = ""
    try:
        image_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
        print("tmp file:" + image_file)
        f = open(image_file, 'wb') 
        f.write(data)
        f.close()
    except:
        response = {'message': 'invalid image'}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")
    print("compare {} and {}".format(image_file, ground_truth))
    draws = match.compare_two('akaze', image_file, ground_truth)
    response = json.dumps(draws)
    print(response)
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/api/v1/add_image', methods=['POST'])
@require_appkey
def add_image():
    json_str = request.json
    json_out = json.loads(json_str)
    
    if json_out['image'] is None or json_out['name'] is None:
        response = {'message': 'missing image or name'}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")
    
    image = json_out['image']
    name = json_out['name']

    data = base64.b64decode(image)
    image_file = ""
    try:
        image_file =str(uuid.uuid4()) + '.jpg'
        f = open(image_file, 'wb') 
        f.write(data)
        f.close()
    except:
        response = {'message': 'invalid image'}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")

    save_image(name, image_file)
    
    response = {'message': 'succeed'}
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

    
@app.route('/')
def index_page():
    return 'Welcome'

app.run(host='0.0.0.0',port='443', debug = False/True, ssl_context=context)
