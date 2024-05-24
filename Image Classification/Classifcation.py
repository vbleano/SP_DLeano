try:
    import os
    import cv2 as cv
    import PIL as Image
    import numpy as np
    from numpy import asarray
    import tensorflow as tf

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    from PIL import Image
    from pillow_heif import register_heif_opener
    from tensorflow.keras import  models
    from ReformatTest import ReformatTest
except Exception as e:
    print(e)
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
def classify():
    img_height = 255
    img_width = 255
    batchSize = 32
    DATADIR = os.getcwd() + "/Testing"
    CATEGORIES = ["Anonang", "Bahai", "Bitaog", "Dangula", "Dao", "Falcata", "Golden Shower",
                  "Ilang Ilang", "Ipil ipil", "Kalantas", "Kalumpit", "Kupang", "Lamio", "Lumbang",
                  "Malapapaya", "Mangium", "Mangkono", "Narra", "Palawan Cherry", "Yemane"]
    model = models.load_model("image_classifier.model")
    path = os.getcwd() + "/TestingConv/Lamio"
    # ReformatTest()
    Anonang = [photo for photo in os.listdir(path) ]
    for photo in Anonang:
        img = Image.open(path+"/"+photo)
        numpyIMG = asarray(img)
        numpyIMG = rgb2gray(numpyIMG)
        numpyIMG.resize(img_height,img_width)

        prediction = model.predict(np.array([numpyIMG]) / 255)
        index = np.argmax(prediction)
        print(prediction)
        print(index)
        print(f"prediction is: {CATEGORIES[index]}")

