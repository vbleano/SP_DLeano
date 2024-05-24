try:
    from keras.preprocessing.image import ImageDataGenerator
    from skimage import io
    import os
    import numpy as np
    from PIL import Image
except Exception as e:
    print(e)


def processImg():
    # creates a path to the directory of the images from OS
    DATADIR = os.getcwd() + "/Data"
    # defines the categories for the images
    CATEGORIES = ["Anonang", "Bahai", "Bitaog", "Dangula", "Dao", "Falcata", "Golden Shower",
                  "Ilang Ilang", "Ipil ipil", "Kalantas", "Kalumpit", "Kupang", "Lamio", "Lumbang",
                  "Malapapaya", "Mangium", "Mangkono", "Narra", "Palawan Cherry", "Yemane"]
    # Defines the path to the converted jpg images
    for i in CATEGORIES:
        path = DATADIR + "/" + i
        # end of defining the path
        # sets the parameters for augmentation
        datagen1 = ImageDataGenerator(
            width_shift_range=0.1,
            fill_mode='constant', cval=125)
        datagen2 = ImageDataGenerator(
            height_shift_range=0.1,
            fill_mode='constant', cval=125)
        datagen3 = ImageDataGenerator(
            horizontal_flip=True,
            fill_mode='constant', cval=125)
        datagen4 = ImageDataGenerator(
            vertical_flip=True,
            fill_mode='constant', cval=125)
        datagen5 = ImageDataGenerator(
            rotation_range=15,
            fill_mode='constant', cval=125)

        dataset = []
        my_images = os.listdir(path)
        for j, image_name in enumerate(my_images):
            if(image_name.split('.')[1] =='jpg'):
                image = io.imread(path+"/"+image_name)
                image = Image.fromarray(image,'RGB')
                image = image.resize((256,256))
                dataset.append(np.array(image))

        x = np.array(dataset)
        k = 0
        # Start of data augmentation
        augPath = DATADIR+"/"+i
        for batch1 in datagen1.flow(x,batch_size=16,
                                  save_to_dir=augPath,
                                  save_prefix='aug1',
                                  save_format='jpg'):
            k+=1
            if k>5:
                break
        k = 0
        for batch2 in datagen2.flow(x,batch_size=16,
                                  save_to_dir=augPath,
                                  save_prefix='aug2',
                                  save_format='jpg'):
            k+=1
            if k>5:
                break
        k = 0
        for batch3 in datagen3.flow(x,batch_size=16,
                                  save_to_dir=augPath,
                                  save_prefix='aug3',
                                  save_format='jpg'):
            k+=1
            if k>5:
                break
        k = 0
        for batch4 in datagen4.flow(x,batch_size=16,
                                  save_to_dir=augPath,
                                  save_prefix='aug4',
                                  save_format='jpg'):
            k+=1
            if k>5:
                break
        k = 0
        for batch5 in datagen5.flow(x,batch_size=16,
                                  save_to_dir=augPath,
                                  save_prefix='aug5',
                                  save_format='jpg'):
            k+=1
            if k>5:
                break

