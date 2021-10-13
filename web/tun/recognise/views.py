# Django libs
from django.shortcuts import render
from django.conf import settings
from django.http import StreamingHttpResponse

#Libraries of ourselves
from .form import ImageForm
from .models import Image
from .camera import VideoCamera
from .constants import BASE_URL, MARKET, SA, SA_D, ST_D, HEADER
from .utilities import egmap, process_json, predict_image, model

# Standard Libraries
import json
import logging

# 3rd party libraries
import requests as rq
import PIL
import numpy as np


res = ""  #result that we will pass to the result page

logger = logging.getLogger(__name__)

# Create your views here.

def home(request):
    return render(request, 'home.html', context={})


def form(request):
    global SA
    global model
    modified_image = Image()
    result_dic = {}
    genre=""
    img=""

    if request.method=="POST":
        form=ImageForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            logger.debug("Image saved")
            obj=form.instance
            logger.info(f"obj is {obj}")
            name_image= obj.image.name
            test_image = obj.image
            target_image = PIL.Image.open(str(settings.BASE_DIR) + obj.image.url)
            logger.info(f"{type(target_image)}")
            image_array = np.array(target_image)
            image_file, x1 = predict_image(image_array, name_image)
            logger.info(f"Image file type: {type(image_file)}")
            modified_image.image = image_file
            logger.debug("next step")
            modified_image.save()
            genre = egmap(x1)
            logger.info(f"genre is {genre}")
            if genre != '':
                SA = SA_D[genre]
                ST = ST_D[genre]
            url = BASE_URL+'?limit=14&market='+MARKET+'&seed_artists='+SA+'&seed_genres='+genre+'&seed_tracks='+ST
            r = rq.get(url, headers=HEADER)
            logger.info(f"{r.status_code}")

            if r.json() and r.status_code == 200:
                json_text = json.loads(r.text)
                result_dic = process_json(json_text["tracks"])
            else:
                logger.warning("Bad Request")
            
            logger.info(f"result is {result_dic}")

            return render(request, "predict.html", {"obj":obj, "prediction":x1, "modified_image":modified_image, "result_dic": result_dic})
        
    
    else:
        form = ImageForm()
        logger.debug("It came into else")
        img = Image.objects.last()
        logger.info(f"Image is {img}")

    return render(request, "predict.html", {"img":img,"form":form})

# def gen(camera):
#     while True:
#         frame = cam.get_frame()
#         # print(frame)
#         m_image, lab =predict_video(frame, "result")
#         print("This is in gen")
#         SA = SA_D[lab]
#         ST = ST_D[lab]
#         print(lab)
#         # m_image = cv2.cvtColor(m_image, cv2.COLOR_RGB2BGR)
#         ret, m_image = cv2.imencode('.jpg', m_image)
#         m_image = m_image.tobytes()
#         yield(b'--frame\r\n'
#               b'Content-Type: image/jpeg\r\n\r\n' + m_image + b'\r\n\r\n')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' +
              frame + b'\r\n\r\n'
        )


def livefeed(request):
    try:
        return StreamingHttpResponse(gen(()), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:  # This is bad! replace it with proper handling
        logger.error(e)

def showlive(request):
    submitb = request.POST.get('submit')
    logger.info(submitb)
    context = {'submitb':submitb} if submitb else {'submitb': None}
    return render(request, 'live.html', context)



def liveres(request):
    logger.debug("This is live res")
    genre = egmap(res)
    logger.info("genre is {}".format(genre))
    if genre != '':
        SA = SA_D[genre]
        ST = ST_D[genre]
    url = BASE_URL+'?limit=7&market='+MARKET+'&seed_artists='+SA+'&seed_genres='+genre+'&seed_tracks='+ST
    r = rq.get(url, headers=HEADER)
    if r.status_code == 200 and r.json():
       json_text = json.loads(r.text)
       result_dic = process_json(json_text["tracks"])
    context = {'result_dic': result_dic, 'emotion': res}
    return render(request, 'liveres.html', context)
    
