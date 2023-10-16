# Facebook Marketplace Recommendation System

This project successfully implements a recommendation system for Facebook Marketplace utilizing a combination of image and tabular data, as well as image similarity search.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Cleaning](#data-cleaning)
- [Model Training](#model-training)
- [Feature Extraction](#feature-extraction)
- [Image Similarity Search](#image-similarity-search)

## Environment Setup

To set up the environment for this project, follow these steps:

1. Clone this repository:
   
```python
git clone https://github.com/PelumiAdeboye/facebook-marketplaces-recommendation-ranking-system
```
   
2. Install the required Python packages:
   
```python
pip install -r requirements.txt
```
   
3. Access your AWS EC2 instance and S3 bucket to download data and files needed for the project.

## Data Cleaning

### Cleaning the Tabular Dataset

Data cleaning for the tabular dataset is complete. The following tasks have been executed:

1. A Python script, `clean_tabular_data.py`, has been created to clean the tabular dataset.

2. All null values in any column have been removed.

3. Prices have been converted into a numerical format by removing pound signs and commas.

4. The main category of each product has been extracted, and labels have been assigned.

### Cleaning the Image Dataset

Image data cleaning has been successfully carried out:

1. A Python script, `clean_images.py`, was created to standardize image sizes and channels.

2. An image-cleaning pipeline was established to ensure consistency in image size and channels.

## Model Training

Model training has been completed. The following objectives have been met:

1. Machine learning models for tabular and image data have been trained in (`classify_images.py`)

2. A dataset (`dataset.py`) forfeeding entries to the model has been created.

3. Transfer learning was employed to fine-tune a pre-trained model (ResNet-50) (`ResNet50_CNN.py`) for image classification.

4. Model weights and label encoder/decoder have been saved in `image_modelDav.pt`.

5. A training loop and validation process were implemented successfully. (`classify_images.py`)

## Feature Extraction

Feature extraction has been accomplished:

1. Image embeddings for every image in the training dataset were extracted using a feature extraction model in (`image_embeddings.py`)

2. A dictionary was created, mapping image IDs to their respective image embeddings. This dictionary was saved as a JSON file named `image_embeddings.json`.

## Image Similarity Search

The image similarity search system is in place:

1. The saved dictionary of image embeddings was loaded.

2. A FAISS model was created with image IDs as the index and corresponding image embeddings as values in file (`faiss1234.py`)

3. An API was implemented to perform vector search for similar images using FAISS in file (`api.py`)

For detailed information and instructions related to each milestone, refer to the relevant section in the project documentation.

## Project Dependencies

- A list of project dependencies is available in the `requirements.txt` file.

## Usage

The project is ready for use, and instructions on how to run and utilize it can be found in the project documentation.

