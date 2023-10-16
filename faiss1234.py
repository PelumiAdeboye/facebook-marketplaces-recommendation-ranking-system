import json
import numpy as np
import faiss
import pickle

# Load the image embeddings from the JSON file
with open("image_embeddings.json", "r") as f:
    image_embeddings = json.load(f)


# Extract image IDs and embeddings
image_ids = list(image_embeddings.keys())
embeddings = list(image_embeddings.values())
# embeddings = [image_embeddings[image_id] for image_id in image_ids]

# Convert embeddings to a NumPy array
embeddings_array = np.array(embeddings, dtype='float32')


# Create a FAISS index
index = faiss.IndexFlatL2(1000)  # You can adjust the parameters
matrix = np.empty([0, 1000])
for key in image_embeddings.keys():
    matrix = np.vstack(
        [matrix, np.array(image_embeddings[key], dtype=np.float32)])

index.add(matrix)
# Add the embeddings to the index
#index.add(embeddings_array)

#saving the Faiss index
faiss.write_index(index, 'index2.pkl')

#saving the list of IDs 
with open('file_ids2.pkl', 'wb') as f:                 
          pickle.dump(image_ids, f)



# Example: Query for similar images based on a given image ID
# image ID you want to search for
query_image_id = "0aa192ee-da53-4800-a56a-d75fab91225a"
n_neighbors = 10        # Number of similar images to retrieve

# Retrieve the embedding of the query image using its ID
if query_image_id in image_embeddings:
    query_embedding = image_embeddings[query_image_id]

    # Perform a vector search
    D, I = index.search(
        np.array(query_embedding, dtype=np.float32), n_neighbors)

    # D contains the distances, and I contains the indices of the most similar images
    # You can use these indices to retrieve the image IDs of the similar images
    similar_image_ids = [image_ids[i] for i in I[0]]
    print("Similar Image IDs:", similar_image_ids)
else:
    print(f"Image ID '{query_image_id}' not found in the dataset.")
