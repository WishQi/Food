from flask import Flask
from flask import request
from src import food_calorie
from flask import jsonify
from src import photo_score

app = Flask(__name__)

def handle_image(image_data):
    return food_calorie.handle_image(image_data)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/api/v1.0/calories', methods=['POST'])
def post_calories():
    if not request.json or not 'title' in request.json:
        assert(400)
    image_data = request.form['image']
    food_image_score = photo_score.score_image(image_data)
    food_image_data = handle_image(image_data)
    food_image_data['score'] = food_image_score
    return jsonify(food_image_data)

if __name__ == '__main__':
    app.run(debug=True)

