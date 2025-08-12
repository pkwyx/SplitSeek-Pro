import os
import json
import warnings
import numpy as np
import pandas as pd

import torch
import argparse

from model_utils import SplitSeek, ESM_Model, featurize, extract_input_data
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from tqdm import tqdm

warnings.filterwarnings("ignore")   

res_d = {
    "GLY": "G", "ALA": "A", "VAL": "V", "LEU": "L", "ILE": "I", 
    "PRO": "P", "PHE": "F", "TYR": "Y", "TRP": "W", "SER": "S", 
    "THR": "T", "CYS": "C", "MET": "M", "ASN": "N", "GLN": "Q", 
    "ASP": "D", "GLU": "E", "LYS": "K", "ARG": "R", "HIS": "H", "MSE": "M",
}

standard_residues = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
}

def is_residues(resname):

    return resname in standard_residues or resname in {
        'SEP', 'TPO', 'PTR', 'TYS', 'CYX',
        'MSE', 'HIP', 'HID', 'HIE', 'HSD',
        'HSE', 'HSP', 'ASH', 'GLH', 'LYN',
        'ARN', 'CYM', 'CYD', 'CY1', 'CSD'
    }

def remove_non_residues(model):

    for chain in list(model):
        residues_to_remove = []
        for residue in chain:
            if not is_residues(residue.get_resname()):
                residues_to_remove.append(residue.id)

        for res_id in residues_to_remove:
            chain.detach_child(res_id)

def parse_pdb(path):

    name = os.path.split(path)[1].split('.')[0]

    if path.endswith('pdb'):

        p = PDBParser()
        s = p.get_structure(name, path)[0]

    elif path.endswith('cif'):

        c = MMCIFParser()
        s = c.get_structure(name, path)[0]

    else:
        print('please input appropriate format')
        return 
    
    target_atoms = ['N', 'CA', 'C', 'O']
    info_d = {}
    length = 0
    remove_non_residues(s)  

    for chain in s.get_chains():

        chain_seq = ''
        chain_coords = []
        chain_calres = []
        chain_id = chain.id

        for res in chain.get_residues():

            rn = res.get_resname()
            if rn in res_d:

                chain_calres.append(res)
                chain_seq += res_d[rn]
                res_xyz = []

                for atom_name in target_atoms:
                    if res.has_id(atom_name):
                        atom = res[atom_name]
                        xyz = list(atom.get_coord())
                        res_xyz.append(xyz)

                    else:
                        xyz = [np.nan, np.nan, np.nan]
                        res_xyz.append(xyz)

                chain_coords.append(res_xyz)

        info_d[chain_id] = [chain_seq, chain_coords, chain_calres]
        length += len(chain_seq)

    return name, info_d, length, s

def rewrite_pdb(pred_list, cal_res, structure, out_path):

    io = PDBIO()
    chain_list = structure.get_chains()
    sc_d = {}
    c = 0

    for chain, cr_chain in zip(chain_list, cal_res):

        for residue in chain.get_residues():
            
            res_info = ''.join([residue.get_resname(), str(residue.id[1])])
            res_info = '_'.join([chain.id, res_info])

            if residue in cr_chain:
                score = pred_list[c] 
                c += 1
                for atom in residue.get_atoms():
                    atom.set_bfactor(score * 100)

            else:
                score = 0.5
                for atom in residue.get_atoms():
                    atom.set_bfactor(50)

            sc_d[res_info] = score

    io.set_structure(structure)
    io.save(out_path)

    return sc_d

def build_model(model_name):

    hidden_dim = 128
    num_encoder_layers = 3
    num_decoder_layers = 3
    num_neighbors = 48
    dropout = 0.1
    backbone_noise = 0
    aaindex1 = 'constant/aaindex.json'
    with open(aaindex1, 'r') as f:
        aaindex1 = json.load(f)

    weight_path = f'weights/{model_name}.pt'
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    model = SplitSeek(aaindex1=aaindex1,
                node_features=hidden_dim,
                edge_features=hidden_dim,
                hidden_dim=hidden_dim,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                k_neighbors=num_neighbors,
                dropout=dropout,
                augment_eps=backbone_noise)
    
    model.to(device)
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    esm_model = ESM_Model()
    esm_model.load("esm2_t33_650M_UR50D")

    return model, esm_model


def run_splitseek(full_info_d, model, esm_model):

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    input_list = extract_input_data(full_info_d)
    X, S, ESM_S, mask, residue_idx, chain_encoding_all, batch_name = featurize(input_list, esm_model, device)
    probs_list = []
    for _ in range(10):
        probs = model(X, S, ESM_S, mask, residue_idx, chain_encoding_all)
        probs_list.append(probs)
    
    probs = torch.mean(torch.stack(probs_list), dim=0)
    probs = probs.view(probs.shape[0], probs.shape[1])
    probs = probs.cpu().data.numpy()

    return probs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Split Sites Prediction',
        formatter_class=argparse.RawTextHelpFormatter
        )

    parser.add_argument('-i', '--input', default=None, help = 'Input PDB file path or dir path, required.')
    parser.add_argument('-o', '--output', default=None, help = 'Output path, optional.')
    # parser.add_argument('-c', '--chainID', default=None, help = 'The chain that you want to design, default all.')
    parser.add_argument('-m', '--model', help = 'which model weights.', default='weights_v1')

    args = parser.parse_args()

    if not args.input:
        quit()

    else:

        if os.path.isdir(args.input):
            input_pdb_list = [os.path.join(args.input, i) for i in os.listdir(args.input) if i.endswith('pdb') or i.endswith('cif')]
            input_dir = args.input

        else:
            input_pdb_list = [args.input]
            input_dir = os.path.split(args.input)[0]

    model_name = args.model

    if not args.output:
        work_dir = input_dir

    else:
        work_dir = args.output
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

    splitmpnn, esm = build_model(model_name)
  
    for pdb in tqdm(input_pdb_list):
            
        try:
            full_info_d, length_d, structure_d = {}, {}, {}
            name, info_d, length, s = parse_pdb(pdb)
            full_info_d[name] = info_d
            probs = run_splitseek(full_info_d, splitmpnn, esm)[0]
            cal_res = [i[2] for i in info_d.values()]

            out_name = '_'.join([name, 'pred'])
            out_pdb = '.'.join([out_name, 'pdb'])
            out_csv = '.'.join([out_name, 'csv'])
            pdb_path = os.path.join(work_dir, out_pdb)
            csv_path = os.path.join(work_dir, out_csv)

            sc_info = rewrite_pdb(probs, cal_res, s, pdb_path)
            df = pd.DataFrame.from_dict(sc_info, orient='index', columns=['score'])
            df.to_csv(csv_path)

        except:
            print('Error in predicting: {}'.format(pdb))
            continue
