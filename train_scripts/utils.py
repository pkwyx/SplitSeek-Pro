import numpy as np
import os
import torch
import random
import json
import time

def worker_init_fn(worker_id):
    np.random.seed()

def build_training_clusters(params, debug):
    val_ids = [l.replace('\n', '') for l in open(params['VAL']).readlines()]
    test_ids = [l.replace('\n', '') for l in open(params['TEST']).readlines()]
    train_ids = [l.replace('\n', '') for l in open(params['TRAIN']).readlines()]

    if debug:
        val_ids = []
        test_ids = []
        train_ids = []

    return train_ids, val_ids, test_ids


def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000):
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 
                     'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    c = 0
    c1 = 0
    pdb_dict_list = []
    for _ in range(repeat):
        for step, t in enumerate(data_loader):
            t = {k:v[0] for k,v in t.items()}
            c1 += 1
            if 'name' in list(t):
                my_dict = {}
                s = 0
                concat_seq = ''
                mask_list = []
                visible_list = []
                if len(list(np.unique(t['idx']))) < 352:
                    for idx in list(np.unique(t['idx'])):
                        letter = chain_alphabet[idx]
                        res = np.argwhere(t['idx']==idx)
                        initial_sequence = "".join(list(np.array(list(t['seq']))[res][0,]))
                        seq_emb = t['seq_emb']
                        emb_idx = 0
                        labels = np.array(t['label'])

                        if initial_sequence[-6:] == "HHHHHH":
                            res = res[:, :-6]
                            labels = labels[:-6]
                            emb_idx = -6
                        if initial_sequence[0:6] == "HHHHHH":
                            res = res[:,6:]
                            labels = labels[6:]
                            emb_idx = 6
                        if initial_sequence[-7:-1] == "HHHHHH":
                            res = res[:, :-7]
                            labels = labels[:-7]
                            emb_idx = -7
                        if initial_sequence[-8:-2] == "HHHHHH":
                            res = res[:,:-8]
                            labels = labels[:-8]
                            emb_idx = -8
                        if initial_sequence[-9:-3] == "HHHHHH":
                            res = res[:,:-9]
                            labels = labels[:-9]
                            emb_idx = -9
                        if initial_sequence[-10:-4] == "HHHHHH":
                            res = res[:,:-10]
                            labels = labels[:-10]
                            emb_idx = -10
                        if initial_sequence[1:7] == "HHHHHH":
                            res = res[:,7:]
                            labels = labels[7:]
                            emb_idx = 7
                        if initial_sequence[2:8] == "HHHHHH":
                            res = res[:,8:]
                            labels = labels[8:]
                            emb_idx = 8
                        if initial_sequence[3:9] == "HHHHHH":
                            res = res[:,9:]
                            labels = labels[9:]
                            emb_idx = 9
                        if initial_sequence[4:10] == "HHHHHH":
                            res = res[:,10:]
                            labels = labels[10:]
                            emb_idx = 10

                        if res.shape[1] < 4:
                            pass
                        else:
                            my_dict['seq_chain_' + letter] = "".join(list(np.array(list(t['seq']))[res][0,]))
                            concat_seq += my_dict['seq_chain_' + letter]
                            if idx in t['masked']:
                                mask_list.append(letter)
                            else:
                                visible_list.append(letter)

                            coords_dict_chain = {}
                            all_atoms = np.array(t['xyz'][res,])[0,]
                            coords_dict_chain['N_chain_'+letter] = all_atoms[:,0,:].tolist()
                            coords_dict_chain['CA_chain_'+letter] = all_atoms[:,1,:].tolist()
                            coords_dict_chain['C_chain_'+letter] = all_atoms[:,2,:].tolist()
                            coords_dict_chain['O_chain_'+letter] = all_atoms[:,3,:].tolist()
                            my_dict['coords_chain_'+letter] = coords_dict_chain
                            my_dict['seq_emb_file_chain_'+letter] = seq_emb
                            my_dict['seq_emb_idx_chain_' +letter] = emb_idx
                            my_dict['probability_'+letter] = labels

                    my_dict['name'] = t['name']
                    my_dict['masked_list'] = mask_list
                    my_dict['visible_list'] = visible_list
                    my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                    my_dict['seq'] = concat_seq
                    my_dict['mask'] = t['mask_label']
                    if len(concat_seq) <= max_length:
                        pdb_dict_list.append(my_dict)
                    
                    if len(pdb_dict_list) >= num_units:
                        break
    
    return pdb_dict_list

def loader_pdb(item, params):

    pdbFile = "%s/cp_pdb/%s.pt"%(params['DIR'], item)
    seqFile = "%s/cp_pdb_esm/%s.pt"%(params['DIR'], item)

    if (not os.path.isfile(pdbFile)) or (not os.path.isfile(seqFile)):
        return {'seq':np.zeros(5)}
    
    chain = torch.load(pdbFile)
    L = len(chain['seq'])
    return {'name'   : item,
            'seq'    : chain['seq'],
            'seq_emb': seqFile,
            'xyz'    : chain['xyz'],
            'idx'    : torch.zeros(L).int(),
            'masked' : torch.Tensor([1]).int(),
            'mask_label' : chain['mask'],
            'label'  : torch.tensor(chain['label'])}
    
class PDBDataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, params):
        self.IDs = IDs
        self.loader = loader
        self.params =params

    def __len__(self):
        return len(self.IDs)
    
    def __getitem__(self, index):
        name = self.IDs[index]
        out = self.loader(name, self.params)
        return out
    
class StructureDataset():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100, alphabet='ACDEFGHIKLMNPQRSTVWYX'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)

                else:
                    discard_count['too_long'] += 1

            else:
                discard_count['bad_chars'] += 1

            if truncate is not None and len(self.data) == truncate:
                return
            
            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
                    
class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True, collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)
        
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size

            else:
                clusters.append(batch)
                batch, batch_max = [], 0

        if len(batch) > 0:
            clusters.append(batch)
        
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)
    
    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5)) * \
            min(step ** (-0.5), step * self.warmup ** (-1.5))
    
    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
        )
