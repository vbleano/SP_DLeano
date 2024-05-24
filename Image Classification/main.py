try:
    from ReformatFile import Reformat
    from imgAugmentation import processImg
    from LoadData import LoadData
    from ModelCreation import trainModel
    from Classifcation import classify
except Exception as e:
    print(e)


# Main Function to be run for classification
if __name__ == '__main__':
    # ============= for image classification =================

    Reformat() #working properly
    # processImg() #working properly
    # Loading the data from the directory
    # training,validation = LoadData()
    # print(training)
    # Creates a model
    # trainModel(training,validation)
    # For classifying
    classify()
