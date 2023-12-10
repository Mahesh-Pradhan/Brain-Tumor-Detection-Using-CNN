import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical.h5')

# model=load_model('BrainTumor10Epochs.h5')
def predict(images):
        
    image=cv2.imread(images)

    img=Image.fromarray(image)

    img=img.resize((64,64))

    img=np.array(img)

    input_img=np.expand_dims(img,axis=0)

    predict_classes= model.predict(input_img)
    result=np.argmax(predict_classes,axis=1)
    return result

print(predict('F:\\Brain Tumour Detection\\pred\\pred5.jpg'))

