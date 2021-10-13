import random
import cv2
import logging
import os

from .constants import IMG_SIZE, HF_PATH, EMOTIONS


from django.core.files.base import ContentFile
from django.core.files import File
from django.conf import settings

from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

model_path = os.path.join(settings.BASE_DIR, '6')
model=load_model(model_path, compile=False)


try:
    HF = cv2.CascadeClassifier(HF_PATH)
except Exception as e:
    logger.error("HF error is {}".format(e))


def egmap(emotionout):
    '''
    link between genre and emotion.
    '''
    print("emotionout is {}".format(emotionout))
    genrechosen=""
    afraidlist = ["hiphop"]
    angrylist = ["rock", "metal"]
    disgustlist = ["hiphop", "jazz"]
    happylist = ["pop", "disco"]
    neutrallist = ["reggae", "classical"]
    sadlist = ["blues", "classical", "country"]
    surprisedlist = ["disco"]
    
    if emotionout == 'afraid':
        genrechosen = random.choice(afraidlist)
    if emotionout == 'angry':
        genrechosen = random.choice(angrylist)
    if emotionout == 'disgust':
        genrechosen = random.choice(disgustlist)
    if emotionout == 'happy':
        genrechosen = random.choice(happylist)
    if emotionout == 'neutral':
        genrechosen = random.choice(neutrallist)
    if emotionout == 'sad':
        genrechosen = random.choice(sadlist)
    if emotionout == 'surprise':
        genrechosen = random.choice(surprisedlist)
    print("genrechosen is {}".format(genrechosen))
    return genrechosen


def prepare(img):
    img_array = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_array=img_array/255.0  
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1,IMG_SIZE, IMG_SIZE,1)


def process_json(json):
    artists_l = []
    songs = []
    urls = []
    s_images = []
    inde = []
    counter = 0
    for entry in json:
        logger.info(f"{entry['artists'][0].keys()}")
        if entry["preview_url"] != None: 
            artists_l.append(entry["artists"][0]["name"])
            songs.append(entry["name"])
            urls.append(entry["preview_url"])
            s_images.append(entry["album"]["images"][0]["url"])
            inde.append(counter)
            counter = counter + 1
    logger.info(f"artists are {artists_l} \n songs are {songs} and urls are {urls}")
    return zip(inde, artists_l, songs, urls, s_images)



def predict_image(image_array, name_image):
    label="Nothing predicted"
    try:
        logger.info(f"Inside predict_image shape: {image_array.shape}")
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        img = image_array.copy()
        faces = HF.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        try:
            logger.debug("Before faces")
            faces = sorted(faces, reverse=True, key = lambda x: (x[2]-x[0]) *(x[3]-x[1]))[0]
            logger.debug("After faces")
            (x,y,w,h)=faces
            logger.debug("Afer co-ordinates are found out")
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(220,40,50),2)
            logger.debug("Before roi is extracted")
            roi = img[y:y+h, x:x+w]
            logger.debug("After roi is extracted")
            logger.info(f"Image shape is {img.shape}")
            prediction = model.predict([prepare(roi)])
            
            preds = prediction[0]
            logger.info(f"prediction is {preds} \n max index is {preds.argmax()}")
            label = EMOTIONS[preds.argmax()]
            logger.info(f"label is {label}")
            cv2.rectangle(img,(x,y+h+10),(x+w,y+h+70),(220,40,50),-2)
            cv2.putText(img,label, (x+10, y+h+50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (225, 225, 225), 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('result.jpeg', img)
        except Exception as e:
            logger.error("Error {e}")
        
        _, buffer_img = cv2.imencode('.jpeg', img)
        f_img = buffer_img.tobytes()
        f1 = ContentFile(f_img)
        image_file = File(f1, name=name_image)
        return image_file, label
    except Exception as e:
        logger.error(f"Error: {e}")


def predict_video(image_array, name_image):
    label="Nothing predicted"
    try:
        print('Inside predict_image shape: {}'.format(image_array.shape))
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        img = image_array.copy()
        faces = HF.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        print("faces is {}".format(faces))
        try:
            print("Before faces")
            faces = sorted(faces, reverse=True, key = lambda x: (x[2]-x[0]) *(x[3]-x[1]))[0]
            print("After faces")
            (x,y,w,h)=faces
            print("After coordinatese")  
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(220,40,50),2)
            print("Before roi")
            roi = img[y:y+h, x:x+w]
            print("Afer roi")
            print('Image shape is {}'.format(img.shape))
            prediction = model.predict([prepare(roi)])
            
            preds = prediction[0]
            print("prediction is {}".format(preds))
            print("max index is {}".format(preds.argmax()))
            label = EMOTIONS[preds.argmax()]
            print("label is {}".format(label))
            cv2.rectangle(img,(x,y+h+10),(x+w,y+h+70),(220,40,50),-2)
            cv2.putText(img,label, (x+10, y+h+50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (225, 225, 225), 3)
            cv2.imwrite('result.jpeg', img)
        except Exception as e:
            logger.error("Something happened during prediction {}".format(e))
        
        _, buffer_img = cv2.imencode('.jpeg', img)
        f_img = buffer_img.tobytes()
        f1 = ContentFile(f_img)
        image_file = File(f1, name=name_image)
        res = label
        logger.info(f"This is res -> {res}")
        return img, label
    except Exception as e:
        logger.error(e)


