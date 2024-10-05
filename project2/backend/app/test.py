
# from backend.app.eda import DataGenerator
from eda import DataGenerator
from load_data import Datasource
from model import Unet
from config import VOLUME_SLICES, VOLUME_START_AT, IMG_SIZE

import os
import matplotlib
import numpy as np

from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


VOLUME_SLICES = 100
VOLUME_START_AT = 22
IMG_SIZE = 128

#  Test of the model 
if __name__ == "__main__":
    
    dataset = os.getenv('DATASET_NAME')
    download_path = os.getenv('DATASET_PATH')
    
    load_dotenv()
    MODELS_DIR = os.getenv('MODELS_DIR')
    
    source = Datasource()
    # source.download_dataset(dataset, download_path)
    source.rename_file()
    source.load_nii_as_narray()
    source.expert_segmentation()
    
    
    # __________________________________________Data Spliting_____________________________________#
    # Split the Dataset
    train_and_test_ids = source.pathListIntoIds()
    source.plot_train_val_test_frequence()
    
    # train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2)
    # train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15)
    
    
    
 
    #________________________________________DataGenerator__________________________________#
    """Utilized to process and send data to the neural network without overloading memory."""
    
    training_generator = DataGenerator(source.train_ids)
    validation_generator = DataGenerator(source.val_ids)
    test_generator = DataGenerator(source.test_ids)
    
    # Retrieve the batch from the training generator
    X_batch, Y_batch = training_generator[8]

    # Extract Flair, T1CE, and segmentation from the batch
    flair_batch = X_batch[:, :, :, 0]
    t1ce_batch = X_batch[:, :, :, 1]
    segmentation_batch = np.argmax(Y_batch, axis=-1)  # Convert one-hot encoded to categorical

    # Extract the 50th slice from Flair, T1CE, and segmentation
    slice_index = 60  # Indexing starts from 0
    slice_flair = flair_batch[slice_index]
    slice_t1ce = t1ce_batch[slice_index]
    slice_segmentation = segmentation_batch[slice_index]

    # Display the 50th slice and its segmentation
    source.display_slice_and_segmentation(slice_flair, slice_t1ce, slice_segmentation)


    #______________________________________Initialisation de la classe Unet_________________#
    """2D U-Net: Faster and requires less memory, advantageous when working with large datasets or limited computational resources."""
    unet = Unet(img_size=IMG_SIZE, num_classes=4)

    # Compilation du modèle
    unet.compile_model()

    # Affichage du modèle sous forme d'image
    # unet.plot_model('unet_model.png')

    # Entraînement du modèle
    # history = unet.train(training_generator, validation_generator, epochs=35, train_ids=source.train_ids)

    # Sauvegarde du modèle entraîné
    # unet.save_model('my_model.keras')

    # Chargement du modèle
    # unet.load_model('my_model.keras')
    
    #___________________________________New model_____________________________________________#
    unet_model= Unet(img_size=IMG_SIZE, num_classes=4)
    # Compile a model and load our saved weights
    unet_model.compile_and_load_weights( os.path.join(MODELS_DIR,'my_model.keras') )
    
    unet_model.showPredictsById(case=source.test_ids[0][-3:])
    
    unet_model.showPredictsById(case=source.test_ids[5][-3:])
    print("Hello false prediction")
    unet_model.predictFromFiles("/Users/omer/Desktop/MLOps_project_group3/project2/dataops/brain_data/BraTSData/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii", "/Users/omer/Desktop/MLOps_project_group3/project2/dataops/brain_data/BraTSData/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii")
    
    print("Segmentations")
    unet_model.show_predicted_segmentations("/Users/omer/Desktop/MLOps_project_group3/project2/dataops/brain_data/BraTSData/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii", "/Users/omer/Desktop/MLOps_project_group3/project2/dataops/brain_data/BraTSData/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii", 60)
    
    cmap = matplotlib.colors.ListedColormap(['#440054', '#3b528b', '#18b880', '#e6d74f'])
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    
    unet_model.show_predicted_segmentations(source.test_ids, 60, cmap, norm)
    
    unet_model.evaluate(test_generator)
    
    
    
    


















