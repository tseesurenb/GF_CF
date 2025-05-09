"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""

import os
import pickle
import glob

import world
import torch
import time
from dataloader import BasicDataset
from torch import nn
import scipy.sparse as sp
import numpy as np
#from sparsesvd import sparsesvd
from scipy.sparse.linalg import svds as sparsesvd
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
    
class LGCN_IDE(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat
        
    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_i = d_mat
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_u = d_mat
        d_mat_u_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsr()
        end = time.time()
        print('training time for LGCN-IDE', end-start)
        
    def getUsersRating(self, batch_users, ds_name):
        norm_adj = self.norm_adj
        batch_test = np.array(norm_adj[batch_users,:].todense())
        U_1 = batch_test @ norm_adj.T @ norm_adj
        if(ds_name == 'gowalla'):
            U_2 = U_1 @ norm_adj.T @ norm_adj
            return U_2
        else:
            return U_1
        
class GF_CF(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat
        
    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        ut, s, self.vt = sparsesvd(self.norm_adj, 256)
        end = time.time()
        print('training time for GF-CF', end-start)
        
    def getUsersRating(self, batch_users, ds_name):
        norm_adj = self.norm_adj
        adj_mat = self.adj_mat
        batch_test = np.array(adj_mat[batch_users,:].todense())
        U_2 = batch_test @ norm_adj.T @ norm_adj
        if(ds_name == 'amazon-book'):
            ret = U_2
        else:
            U_1 = batch_test @  self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
            ret = U_2 + 0.3 * U_1
        return ret


class GF_CF_v2(object):
    def __init__(self, adj_mat, cache_dir='./model_cache'):
        self.adj_mat = adj_mat
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_similarity_matrix_filename(self):
        n_users, n_items = self.adj_mat.shape
        return os.path.join(self.cache_dir, f"item_sim_matrix_{n_users}x{n_items}.pkl")
    
    def _get_normalized_matrix_filename(self, k_i):
        n_users, n_items = self.adj_mat.shape
        return os.path.join(self.cache_dir, f"item_sim_norm_k{k_i}_{n_users}x{n_items}.pkl")
    
    def _file_exists(self, path):
        return os.path.isfile(path)
    
    def _save_matrix(self, path, matrix):
        with open(path, 'wb') as f:
            pickle.dump(matrix, f)
    
    def _load_matrix(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _progress(self, desc, iterable):
        return tqdm(iterable, desc=desc, ncols=80)

    def train(self, k_i=20):
        start = time.time()

        sim_path = self._get_similarity_matrix_filename()
        norm_path = self._get_normalized_matrix_filename(k_i)

        # Step 1: Load or compute item-item similarity
        if self._file_exists(sim_path):
            print("Loading cached item similarity matrix...")
            self.item_sim = self._load_matrix(sim_path)
        else:
            print("Step 1: Computing item-item similarity matrix...")
            item_vectors = self.adj_mat.T.todense()  # item x user
            self.item_sim = cosine_similarity(item_vectors)
            self._save_matrix(sim_path, self.item_sim)
            print("Similarity matrix computed and cached.")

        # Step 2: Top-K filtering
        if self._file_exists(norm_path):
            print("Loading cached normalized matrix...")
            self.item_sim_norm = self._load_matrix(norm_path)
        else:
            print(f"\nStep 2: Applying top-K filtering (k_i = {k_i})...")
            ind = np.argpartition(self.item_sim, -k_i, axis=1)[:, -k_i:]
            mask = np.zeros_like(self.item_sim, dtype=bool)
            rows = np.arange(self.item_sim.shape[0])[:, np.newaxis]
            mask[rows, ind] = True

            filtered_sim = np.zeros_like(self.item_sim)
            filtered_sim[mask] = self.item_sim[mask]
            self.item_sim = filtered_sim
            print(f"Top-{k_i} filtering done. Non-zero entries: {np.count_nonzero(self.item_sim)}")

            # Step 3: Symmetric softmax normalization
            print("\nStep 3: Applying symmetric softmax normalization...")
            exp_sim = np.exp(self.item_sim)
            outgoing_sum = np.sum(exp_sim, axis=1, keepdims=True)
            incoming_sum = np.sum(exp_sim, axis=0, keepdims=True)
            denominator = np.sqrt(outgoing_sum @ incoming_sum)

            zero_mask = (denominator == 0)
            if np.any(zero_mask):
                print(f"Found {np.sum(zero_mask)} zero denominators, replacing with 1.0")
                denominator[zero_mask] = 1.0

            self.item_sim_norm = exp_sim / denominator
            self._save_matrix(norm_path, self.item_sim_norm)
            print("Normalization complete and cached.")

        print(f"\nTraining complete in {time.time() - start:.2f} seconds.")

    def getUsersRating(self, batch_users, ds_name=None):
        user_interactions = self.adj_mat[batch_users, :].todense()
        pred_scores = user_interactions @ self.item_sim_norm
        return pred_scores

    def list_cached_files(self):
        files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        if files:
            print("\nCached files:")
            for f in files:
                print(f"- {os.path.basename(f)}")
        else:
            print("\nNo cached files found.")


class GF_CF_v1(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat
        
    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        
        print("Step 1: Computing item-item similarity matrix...")
        item_vectors = np.array(adj_mat.T.todense())  # Transpose to get item vectors
        #item_vectors = np.asarray(adj_mat.T.todense())

        print(f"Item vectors shape: {item_vectors.shape}")
        
        # Calculate cosine similarity between items
        self.item_sim = cosine_similarity(item_vectors)
        print(f"Similarity matrix shape: {self.item_sim.shape}")
        #print(f"Sample similarity values (first 5x5):\n{self.item_sim[:5, :5]}")
        
        print("\nStep 2: Applying top-K filtering...")
        k_i = 20  # Number of similar items to keep for each item
        print(f"Using k_i = {k_i}")
        
        # Get indices of top k_i values for each row
        ind = np.argpartition(self.item_sim, -k_i, axis=1)[:, -k_i:]
        print(f"Found top-{k_i} indices for each item")
        
        # Create mask with zeros
        mask = np.zeros_like(self.item_sim, dtype=bool)
        # For each row i, set mask[i, ind[i]] to True
        rows = np.arange(self.item_sim.shape[0])[:, np.newaxis]
        mask[rows, ind] = True
        
        # Apply mask
        filtered_sim = np.zeros_like(self.item_sim)
        filtered_sim[mask] = self.item_sim[mask]
        self.item_sim = filtered_sim
        
        print(f"After filtering, non-zero entries: {np.count_nonzero(self.item_sim)}")
        #print(f"Sample filtered similarity values (first 5x5):\n{self.item_sim[:5, :5]}")
        
        print("\nStep 3: Applying symmetric softmax normalization...")
        # First, apply exponential to similarity scores
        exp_sim = np.exp(self.item_sim)
        print(f"Exponential applied, values range: {np.min(exp_sim[exp_sim > 0]):.6f} to {np.max(exp_sim):.6f}")
        
        # Calculate outgoing sum for each item (row sum)
        outgoing_sum = np.sum(exp_sim, axis=1, keepdims=True)
        print(f"Outgoing sums shape: {outgoing_sum.shape}")
        #print(f"Sample outgoing sums (first 5):\n{outgoing_sum[:5].flatten()}")
        
        # Calculate incoming sum for each item (column sum)
        incoming_sum = np.sum(exp_sim, axis=0, keepdims=True)
        print(f"Incoming sums shape: {incoming_sum.shape}")
        #print(f"Sample incoming sums (first 5):\n{incoming_sum[0, :5]}")
        
        # Apply symmetric softmax normalization
        # For each (i,j), exp(sim[i,j]) / sqrt(sum_row_i * sum_col_j)
        denominator = np.sqrt(outgoing_sum @ incoming_sum)
        print(f"Denominator matrix shape: {denominator.shape}")
        #print(f"Sample denominator values (first 5x5):\n{denominator[:5, :5]}")
        
        # Avoid division by zero
        zero_mask = denominator == 0
        if np.any(zero_mask):
            print(f"Found {np.sum(zero_mask)} zero denominators, replacing with 1.0")
            denominator[zero_mask] = 1.0
        
        self.item_sim_norm = exp_sim / denominator
        print(f"Normalized matrix shape: {self.item_sim_norm.shape}")
        #print(f"Sample normalized values (first 5x5):\n{self.item_sim_norm[:5, :5]}")
        print(f"Normalized values range: {np.min(self.item_sim_norm[self.item_sim_norm > 0]):.6f} to {np.max(self.item_sim_norm):.6f}")
        
        end = time.time()
        print(f'\nTraining time for DySimItem: {end-start:.4f} seconds')

    def getUsersRating(self, batch_users, ds_name):
        #print(f"Getting ratings for {len(batch_users)} users...")
        
        # Get user interaction data
        batch_test = np.array(self.adj_mat[batch_users, :].todense())
        #print(f"User interaction matrix shape: {batch_test.shape}")
        
        # Use normalized item-item similarity for prediction (equivalent to U2)
        #print("Computing predictions using normalized item-item similarity matrix...")
        pred_scores = batch_test @ self.item_sim_norm
        #print(f"Prediction scores shape: {pred_scores.shape}")
        #print(f"Prediction values range: {np.min(pred_scores):.6f} to {np.max(pred_scores):.6f}")
        
        return pred_scores