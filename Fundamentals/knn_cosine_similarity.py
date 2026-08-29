import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return dot_product / (norm_vec1 * norm_vec2)


def k_nearest_neighbor(vec, candidates, k=3):
    similarities = []

    for c in candidates:
        cos_similarity = cosine_similarity(c, vec)
        similarities.append(cos_similarity)
        
    # sort by asc
    sorted_ids = np.argsort(similarities)

    # get the last indices as the k most similar
    k_idx = sorted_ids[-k:]
    return k_idx