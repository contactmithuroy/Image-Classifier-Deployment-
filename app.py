from flask import Flask, render_template, request

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
#from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

app = Flask(__name__)
model = ResNet50()

@app.route('/', methods=['GET'])
def templet():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    classification_1 = label[1].split(',')[0].replace('_', ' ').title()
    classification_2 = '%.2f' % (label[2] * 100)

    classification_1_encoded = classification_1.encode('utf-8').decode('utf-8')
    classification_2_encoded = classification_2.encode('utf-8').decode('utf-8')

    return render_template('index.html', prediction=classification_1, classification_2=classification_2, image_path=image_path)




if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
