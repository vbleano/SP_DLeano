try:
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
except Exception as e:
    print(e)
def LoadData():
    # Sets the values for the image size and batch size.
    img_height = 255
    img_width = 255
    batchSize = 32

    # ===================== loading the dataset =========================
    # for training dataset
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        'Data',
        labels='inferred', # sets the labels inferred depending on subdirectory
        label_mode="categorical",
        class_names=["Anonang", "Bahai", "Bitaog","Dangula","Dao","Falcata","Golden Shower",
                     "Ilang Ilang","Ipil ipil","Kalantas","Kalumpit","Kupang","Lamio","Lumbang",
                     "Malapapaya","Mangium","Mangkono","Narra","Palawan Cherry","Yemane"],  # Categories of the data accroding to subdirectory
        color_mode='grayscale',  # sets the images to grayscale
        batch_size=batchSize,  # sets the batch size to 2
        image_size=(img_height, img_width),  # reshape if not in this size
        shuffle=True,  # randomizes the images
        seed=1337,
        validation_split=0.3,  # sets the images to be split into 70-30 (training, validation)
        subset="training"
    )

    # for validation dataset
    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        'Data',
        labels='inferred',
        label_mode="categorical",
        class_names=["Anonang", "Bahai", "Bitaog", "Dangula", "Dao", "Falcata", "Golden Shower",
                     "Ilang Ilang", "Ipil ipil", "Kalantas", "Kalumpit", "Kupang", "Lamio", "Lumbang",
                     "Malapapaya", "Mangium", "Mangkono", "Narra", "Palawan Cherry", "Yemane"],
        color_mode='grayscale',
        batch_size=batchSize,
        image_size=(img_height, img_width),
        seed=1337,
        validation_split=0.3,
        subset="validation"
    )
    return ds_train, ds_validation
    # ===================================================================