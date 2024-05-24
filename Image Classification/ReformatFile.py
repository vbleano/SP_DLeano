# Code for reformatting the raw data from HEIC to JPEG
try:
    import os
    from PIL import Image
    from pillow_heif import register_heif_opener
except Exception as e:
    print(e)
def Reformat():
    # Sets teh directory to raw data
    DATADIR = os.getcwd() + "/Raw_Data"
    # Creates an array for the species of all the seeds to be classified
    CATEGORIES = ["Anonang", "Bahai", "Bitaog","Dangula","Dao","Falcata","Golden Shower",
                     "Ilang Ilang","Ipil ipil","Kalantas","Kalumpit","Kupang","Lamio","Lumbang",
                     "Malapapaya","Mangium","Mangkono","Narra","Palawan Cherry","Yemane"]

    # For Each species in category
    for category in CATEGORIES:
        # Sets the directory per species
        path =  os.path.join(DATADIR, category)
        # Opens a heif opener for reading heic files
        register_heif_opener()
        # Gets all the heic files in the folder
        heic_files = [photo for photo in os.listdir(path) if ".HEIC" in photo]

        # For each heic file in the directory of per species
        for photo in heic_files:
            # sets the heic file in a temp variable
            temp_img = Image.open(path+"/"+photo)
            # Converts the heic file to a jpeg file with the same file name
            jpg_photo = photo.replace(".HEIC",".jpg")
            # Saves the image to a directory
            save_path = os.path.join(os.getcwd() + "/Data")

            # Checks if there is a directory for the images to be saved
            if(not(os.path.isdir(os.getcwd()+"/Data"))):
                # If not, create a directory
                os.mkdir("Data")
                if (os.path.isdir(os.path.join(save_path + '/' + category))):
                    temp_img.save(f"{os.path.join(save_path + '/' + category)}/{jpg_photo}")
                else:
                    os.mkdir(os.path.join(save_path + '/' + category))
                    temp_img.save(f"{os.path.join(save_path + '/' + category)}/{jpg_photo}")
            else:
                if (os.path.isdir(os.path.join(save_path + '/' + category))):
                    temp_img.save(f"{os.path.join(save_path+'/'+category)}/{jpg_photo}")
                else:
                    os.mkdir(os.path.join(save_path + '/' + category))
                    temp_img.save(f"{os.path.join(save_path+'/'+category)}/{jpg_photo}")







