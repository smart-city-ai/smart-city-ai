#!/usr/bin/env python

import requests
import base64
import json

def send_request(img, key, url, name="test"):
    if url is None or img is None:
        return None
    j = {'name': name, "image": img}
    #headers = {'x-api-key': key, 'Content-Type': 'application/json'}
    headers = {'Content-Type': 'application/json'}
    r = requests.post(url, json=json.dumps(j), headers=headers)
    return r

def test_it():
    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['key=', 'url=', 'image=', 'name='])
    opts = dict(opts)
    api_key = opts.get('--key')
    url_addr = opts.get('--url')
    image_file = opts.get('--image')
    name = opts.get('--name')
    print(api_key, url_addr)
    image = open(image_file, 'rb') 
    image_data = image.read()
    image_str = base64.b64encode(image_data).decode('utf-8')
    
    r = send_request(image_str, key=api_key, url = url_addr, name=name)
    print(r.text)



if __name__ == '__main__':
    test_it()
