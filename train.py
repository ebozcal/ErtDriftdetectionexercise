import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample embeddings (replace this with your actual embeddings)
embeddings = np.random.rand(64, 10)

# Set the cosine similarity threshold (e.g., 0.80)
threshold = 0.80

# Initialize an empty dictionary to store the class embeddings
class_embeddings = {}

# Iterate through each embedding vector
for i, embedding in enumerate(embeddings):
    found_class = False

    # Iterate through existing class embeddings
    for class_idx, class_emb_list in class_embeddings.items():
        # Calculate cosine similarity between the current embedding and the class embeddings
        similarities = cosine_similarity([embedding], embeddings[class_emb_list]).flatten()

        # Check if any of the similarities are above the threshold
        if any(similarity >= threshold for similarity in similarities):
            # Add the current embedding to the class
            class_emb_list.append(i)
            found_class = True
            break

    # If the embedding doesn't belong to any existing class, create a new class
    if not found_class:
        class_embeddings[len(class_embeddings)] = [i]

# Print the resulting class dictionary
print(class_embeddings)
