from clarifai.rest import ClarifaiApp
import requests


def food_recognition(image_base64):
    app = ClarifaiApp(api_key='e985c887ce494b8aa1fa2c1b884c34b9')
    model = app.models.get('food-items-v1.0')
    image_base64_byte = bytes(image_base64, encoding='utf-8')
    res = model.predict_by_base64(base64_bytes=image_base64_byte)
    return res['outputs'][0]['data']['concepts']

def food_nultrients(food_name):
    food_name.replace(' ', '%20')
    api_str = 'https://api.edamam.com/api/food-database/parser'
    parameters = {
        "app_id" : "e17fd6eb",
        "app_key" : "a042a4a9a2f3bed58c7e7b04b678b353",
        "ingr" : food_name
    }
    res = requests.get(api_str, params=parameters)
    res = res.json()
    return res['hints'][0]['food']['nutrients']

def handle_image(image_base64):
    concepts = food_recognition(image_base64)
    data_list = []
    for i in range(3):
        concept = concepts[i]
        food_name = concept['name']
        nultrients = food_nultrients(food_name)
        data = {
            'name' : food_name,
            'possibility' : concept['value'],
            'nultrients' : nultrients
        }
        data_list.append(data)
    return {'data' : data_list}