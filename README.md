# Facebook Marketplace Recommendation Ranking System

This project aims to build a recommendation system for Facebook Marketplace using a combination of image and tabular data, as well as image similarity search.

## Table of Contents

- [Milestone 1: Set up the Environment](#milestone-1-set-up-the-environment)
- [Milestone 2: An Overview of the System](#milestone-2-an-overview-of-the-system)
- [Milestone 3: Data Cleaning](#milestone-3-data-cleaning)
- [Milestone 4: Training Models](#milestone-4-training-models)
- [Milestone 5: Feature Extraction](#milestone-5-feature-extraction)
- [Milestone 6: Image Similarity Search](#milestone-6-image-similarity-search)

## Milestone 1: Set up the Environment

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

1. Create a Python script, `clean_tabular_data.py`, to clean the tabular dataset.
2. Remove null values in any column.
3. Convert prices into a numerical format by removing pound signs and commas.
4. Extract the main category of the product and assign a label to each entry.

### Cleaning the Image Dataset

1. Create a Python script, `clean_images.py`, to standardize image sizes and channels.
2. Create an image cleaning pipeline to ensure all images are consistent in size and channels.

## Training Models

1. Train machine learning models for tabular and image data.
2. Create a dataset that feeds entries to the model.
3. Use transfer learning to fine-tune a pre-trained model (e.g., ResNet-50) for image classification.
4. Save model weights and the label encoder/decoder.
5. Implement a training loop and validation process.

## Feature Extraction

1. Extract image embeddings for every image in the training dataset using a feature extraction model.
2. Create a dictionary with image embeddings (key: image id, value: image embedding).
3. Save this dictionary as a JSON file named `image_embeddings.json`.

## Image Similarity Search

1. Load the saved dictionary of image embeddings.
2. Create a FAISS model with image ids as the index and corresponding image embeddings as values.
3. Implement an API to perform vector search for similar images using FAISS.

Refer to each milestone in the project documentation for more detailed information and instructions.

## Project Dependencies

- Insert a section here listing all the project's dependencies.

## Usage

Describe how to run and use the project.

## Contributing

- Explain how others can contribute to this project, such as reporting issues or submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

