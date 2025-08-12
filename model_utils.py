import torch
import numpy as np
import random
import json

import torch.nn as nn
import torch.nn.functional as F

def extract_input_data(full_info_d):

    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 
                     'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    full_dict_list = []

    for name, pdb_info in full_info_d.items():

        fseq = ''
        chain_list = []
        my_dict = {}

        for k, i in enumerate(pdb_info):

            coords_dict_chain = {}

            chain_idx = chain_alphabet[k]
            chain_list.append(chain_idx)

            seq, coords = pdb_info[i][0], pdb_info[i][1]
            coords = np.array(coords)

            seq = seq.replace("X", "M")
            fseq += seq
            my_dict['seq_chain_' + chain_idx] = seq
            coords_dict_chain['N_chain_' + chain_idx] = coords[:,0,:].tolist()
            coords_dict_chain['CA_chain_' + chain_idx] = coords[:,1,:].tolist()
            coords_dict_chain['C_chain_' + chain_idx] = coords[:,2,:].tolist()
            coords_dict_chain['O_chain_' + chain_idx] = coords[:,3,:].tolist()
            my_dict['coords_chain_' + chain_idx] = coords_dict_chain

        my_dict['name'] = name
        my_dict['seq'] = fseq
        my_dict['chain_list'] = chain_list
        my_dict['num_of_chains'] =  len(chain_list)

        full_dict_list.append(my_dict)

    return full_dict_list


def featurize(batch, esm, device):

    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    S = np.zeros([B, L_max], dtype=np.int32)
    ESM_S = np.zeros([B, L_max, 1280], dtype=np.float32)

    batch_name = []
    for i,b in enumerate(batch):
        name = b['name']
        batch_name.append(name)
        all_chains = b['chain_list']

        x_chain_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0

        for step, letter in enumerate(all_chains):

            chain_seq = b[f'seq_chain_{letter}']
            chain_length = len(chain_seq)

            esm_emb = esm.encode(chain_seq)[1:-1]

            chain_coords = b[f'coords_chain_{letter}']
            x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1)
            x_chain_list.append(x_chain)

            chain_seq_list.append(chain_seq)
            chain_encoding_list.append(c*np.ones(chain_length))
            l1 += chain_length
            ESM_S[i, l0:l1, :] = esm_emb[:,:]

            l0 += chain_length
            c += 1

        x = np.concatenate(x_chain_list, 0)
        all_sequence = "".join(chain_seq_list)
        chain_encoding = np.concatenate(chain_encoding_list, 0)

        l = len(all_sequence)
        x_pad = np.pad(x, [[0, L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan,))
        X[i,:,:,:] = x_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0,))
        chain_encoding_all[i,:] = chain_encoding_pad
        
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.

    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,device=device)
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    ESM_S = torch.from_numpy(ESM_S).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)

    return X, S, ESM_S, mask, residue_idx, chain_encoding_all, batch_name

def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]

    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1)) # [B,N,K] => [B,N,K,C]
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]

    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))  # [B,N,K] => [B,NK]
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2)) # [B,NK] => [B,NK,C]

    neighbor_features = torch.gather(nodes, 1, neighbors_flat) 
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1]) # [B,NK,C] => [B,N,K,C]
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1,-1,nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    # h_nodes: [Batch, Sequence_length, Hidden_dim]
    # h_neighbors: [Batch, Sequence_length, Number_connection(48), Hidden_dim]
    # E_idx: [Batch, Sequence_length, Number_connection(48)]
    h_nodes = gather_nodes(h_nodes, E_idx)  # [B,N,K,C]
    h_nn = torch.cat([h_neighbors, h_nodes], -1) # [B,N,K,2C]
    return h_nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout = 0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):

        # h_V: [Batch, Sequence_length, Hidden_dim]
        # h_E: [Batch, Sequence_length, Number_connection(48), Hidden_dim]
        # E_idx: [Batch, Sequence_length, Number_connection(48)]

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)   # [B,N,K,2C]  
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)  # [B,N,K,C]  
        h_EV = torch.cat([h_V_expand, h_EV], -1)  # [B,N,K,3C] 
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))  # [B,N,K,C] 

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message  # [B,N,K,C]

        dh = torch.sum(h_message, -2) / self.scale  # [B,N,C] 
        h_V = self.norm1(h_V + self.dropout1(dh))  # [B,N,C] 

        dh = self.dense(h_V)   # [B,N,C]
        h_V = self.norm2(h_V + self.dropout2(dh))  

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)  # [B,N,K,2C]
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)  # [B,N,K,C]
        h_EV = torch.cat([h_V_expand, h_EV], -1)  # [B,N,K,3C]
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))  # [B,N,K,C]
        h_E = self.norm3(h_E + self.dropout3(h_message))

        return h_V, h_E
    
class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):

        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        return h_V

class PositionalEncodings(nn.Module):
    # 相对位置编码，用64个独热编码表示链内的相对位置（若相隔特别远，都是0或64，不做区分）
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E

class ProteinFeatures(nn.Module):
    def __init__(self, aa_dict, edge_features, node_features, node_in=566, num_positional_embeddings=16, 
                 num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        super(ProteinFeatures, self).__init__()
        self.aa_dict = aa_dict
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.node_in = node_in
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf*25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.node_property_embedding = nn.Linear(node_in, node_features, bias=True)
        self.norm_edges = nn.LayerNorm(edge_features)
        self.norm_property_nodes = nn.LayerNorm(node_features)
        self.node_sequence_embedding = nn.Linear(1280, node_features, bias=True)
        self.norm_sequence_nodes = nn.LayerNorm(node_features)
        self.node_embedding = nn.Linear(2 * node_features, node_features)
        self.norm_nodes = nn.LayerNorm(node_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)  # [B,1,N,3] - [B,N,1,3] = [B,N,N,3] 计算所有坐标点之间的向量差
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)  # 计算向量差距离
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)  # 找到距离最小的k个近邻残基

        return D_neighbors, E_idx
    
    def _aapro(self, S:torch.Tensor):
        device = S.device
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        V_init = np.zeros((S.shape[0], S.shape[1], self.node_in))
        for i,seq in enumerate(S):
            for j,aa in enumerate(seq):
                if aa != 0:
                    aapro = self.aa_dict[alphabet[int(aa)]]
                    V_init[i,j,:] = np.array(aapro)

        V_init = torch.from_numpy(V_init).to(dtype=torch.float32, device=device)

        return V_init

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        return RBF
    
    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2, -1) + 1e-6) # [B,N,N]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] # [B,N,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)

        return RBF_A_B
    
    def forward(self, X, S, V_S, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:,:,1,:] - X[:,:,0,:]  # CA - N
        c = X[:,:,2,:] - X[:,:,1,:]  # C - CA
        a = torch.cross(b,c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]        

        D_neighbors, E_idx = self._dist(Ca, mask)
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, N, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)  # 相对位置编码
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        V = self._aapro(S)
        V = self.node_property_embedding(V)
        V = self.norm_property_nodes(V)

        V_S = self.node_sequence_embedding(V_S)
        V_S = self.norm_sequence_nodes(V_S)
        V = torch.cat((V, V_S), -1) 
        V = self.node_embedding(V)
        V = self.norm_nodes(V)

        return V, E, E_idx

class SplitSeek(nn.Module):
    def __init__(self, aaindex1, num_letters=21, node_features=128, edge_features=128, 
                 hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                 vocab=21, k_neighbors=48, augment_eps=0.1, dropout=0.1):
        super(SplitSeek, self).__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(aaindex1, edge_features, node_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_h = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        self.econder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        self.W_out1 = nn.Linear(hidden_dim, int(hidden_dim/2), bias=True)
        self.W_out2 = nn.Linear(int(hidden_dim/2), 1, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, ESM_S, mask, residue_idx, chain_encoding_all):

        V, E, E_idx = self.features(X, S, ESM_S, mask, residue_idx, chain_encoding_all)
        h_V = self.W_h(V)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.econder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_s(S)

        h_ES_encoder = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_ESV_encoder = cat_neighbors_nodes(h_V, h_ES_encoder, E_idx)

        for layer in self.decoder_layers:
            h_V = layer(h_V, h_ESV_encoder, mask)

        h_V = self.W_out1(h_V)
        logits = self.W_out2(h_V)
        probs = F.sigmoid(logits)

        return probs

class ESM_Model:
    # esm1b_t33_650M_UR50S
    # esm2_t6_8M_UR50D
    # esm2_t12_35M_UR50D
    # esm2_t30_150M_UR50D
    # esm2_t33_650M_UR50D
    # esm2_t36_3B_UR50D
    
    def __init__(self, *args):
        if len(args) == 1:
            self.load(args[0])
    
    def load(self, model_name):
        import esm
        self.model_name = model_name
        self.model, alphabet = eval(f'esm.pretrained.{self.model_name}()')
        self.batch_converter = alphabet.get_batch_converter()
        self.model.eval()
        self.embed_dim = self.model._modules['layers'][0].embed_dim
        self.layers = sum(1 for i in self.model._modules['layers'])
        
    def encode(self, sequence, device='cuda', threads=4):
        try:
            torch.cuda.empty_cache()
            torch.set_num_threads(threads)

            batch_labels, batch_strs, batch_tokens = self.batch_converter([['',sequence]])
            batch_tokens = batch_tokens.to(device)
            self.model = self.model.to(device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[self.layers], return_contacts=False)
                results = results["representations"][self.layers].to('cpu')[0]
            return results
        except:
            if device != 'cpu':
                return self.encode(sequence, device='cpu', threads=threads)
            else:
                return None
