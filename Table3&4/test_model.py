import numpy as np
import scipy.sparse as sp
import time

class DySimItemTest(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat
        
    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        
        print("\n=== Step 1: Computing item-item similarity matrix ===")
        item_vectors = np.array(adj_mat.T.todense())  # Transpose to get item vectors
        print(f"Item vectors shape: {item_vectors.shape}")
        print("Item vectors (rows=items, cols=users):")
        print(item_vectors)
        
        # Calculate cosine similarity between items
        from sklearn.metrics.pairwise import cosine_similarity
        self.item_sim = cosine_similarity(item_vectors)
        print(f"\nSimilarity matrix shape: {self.item_sim.shape}")
        print("Item-item similarity matrix:")
        print(np.round(self.item_sim, 3))
        
        print("\n=== Step 2: Applying top-K filtering ===")
        k_i = 3  # Smaller k for our example
        print(f"Using k_i = {k_i}")
        
        # Get indices of top k_i values for each row
        ind = np.argpartition(self.item_sim, -k_i, axis=1)[:, -k_i:]
        print("Top-k indices for each item:")
        print(ind)
        
        # Create mask with zeros
        mask = np.zeros_like(self.item_sim, dtype=bool)
        # For each row i, set mask[i, ind[i]] to True
        rows = np.arange(self.item_sim.shape[0])[:, np.newaxis]
        mask[rows, ind] = True
        
        # Apply mask
        filtered_sim = np.zeros_like(self.item_sim)
        filtered_sim[mask] = self.item_sim[mask]
        self.item_sim = filtered_sim
        
        print(f"\nAfter filtering, non-zero entries: {np.count_nonzero(self.item_sim)}")
        print("Filtered similarity matrix:")
        print(np.round(self.item_sim, 3))
        
        print("\n=== Step 3: Applying symmetric softmax normalization ===")
        # First, apply exponential to similarity scores
        exp_sim = np.exp(self.item_sim)
        print("Exponential of similarity scores:")
        print(np.round(exp_sim, 3))
        
        # Calculate outgoing sum for each item (row sum)
        outgoing_sum = np.sum(exp_sim, axis=1, keepdims=True)
        print("\nOutgoing sums for each item:")
        print(np.round(outgoing_sum, 3))
        
        # Calculate incoming sum for each item (column sum)
        incoming_sum = np.sum(exp_sim, axis=0, keepdims=True)
        print("\nIncoming sums for each item:")
        print(np.round(incoming_sum, 3))
        
        # Apply symmetric softmax normalization
        denominator = np.sqrt(outgoing_sum @ incoming_sum)
        print("\nDenominator matrix (sqrt of outer product):")
        print(np.round(denominator, 3))
        
        # Avoid division by zero
        denominator[denominator == 0] = 1.0
        
        self.item_sim_norm = exp_sim / denominator
        print("\nNormalized similarity matrix:")
        print(np.round(self.item_sim_norm, 5))
        
        end = time.time()
        print(f'\nTraining time: {end-start:.4f} seconds')

    def getUsersRating(self, batch_users):
        print("\n=== Getting user ratings ===")
        
        # Get user interaction data
        batch_test = np.array(self.adj_mat[batch_users, :].todense())
        print(f"Selected user interactions:")
        print(batch_test)
        
        # Use normalized item-item similarity for prediction
        pred_scores = batch_test @ self.item_sim_norm
        print(f"\nPrediction scores:")
        print(np.round(pred_scores, 5))
        
        # Also try the unnormalized version for comparison
        unnorm_pred = batch_test @ self.item_sim
        print(f"\nUnnormalized prediction scores (for comparison):")
        print(np.round(unnorm_pred, 5))
        
        return pred_scores

# Create a small test dataset (5 users x 8 items)
def create_test_data():
    # Create a small interaction matrix
    # 1 indicates the user has interacted with the item
    interaction_matrix = np.array([
        [1, 0, 1, 0, 1, 0, 0, 0],  # User 0
        [0, 1, 1, 0, 0, 1, 0, 0],  # User 1
        [1, 1, 0, 0, 0, 0, 1, 0],  # User 2
        [0, 0, 0, 1, 1, 0, 0, 1],  # User 3
        [0, 0, 1, 1, 0, 0, 1, 0]   # User 4
    ])
    
    # Create a sparse matrix
    return sp.csr_matrix(interaction_matrix)

# Run the test
def run_test():
    # Create test data
    adj_mat = create_test_data()
    print("=== Test Data ===")
    print("User-Item Interaction Matrix (rows=users, cols=items):")
    print(adj_mat.todense())
    
    # Create and train the model
    model = DySimItemTest(adj_mat)
    model.train()
    
    # Get recommendations for all users
    users = np.arange(adj_mat.shape[0])
    pred = model.getUsersRating(users)
    
    # Check top 2 recommendations for each user
    print("\n=== Top 2 Recommendations for Each User ===")
    for user_idx in range(adj_mat.shape[0]):
        # Get items the user hasn't interacted with
        user_interactions = adj_mat[user_idx].toarray().flatten()
        candidate_items = np.where(user_interactions == 0)[0]
        
        # Get scores for these items
        scores = pred[user_idx, candidate_items]
        
        # Sort by score and get top 2
        top_indices = np.argsort(scores)[-2:][::-1]
        recommended_items = candidate_items[top_indices]
        
        print(f"User {user_idx}: Recommended items {recommended_items} with scores {np.round(scores[top_indices], 5)}")

# Run the test
run_test()