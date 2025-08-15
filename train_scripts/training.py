import argparse
import os

def main(args):
    import json, time, os, sys, glob
    import warnings
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import random
    import os.path
    from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, get_std_opt, PDBDataset, StructureLoader, StructureDataset
    from model_utils import featurize, loss_mse, SplitSeek

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    scaler = torch.amp.GradScaler('cuda')
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    data_path = args.path_for_training_data

    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != "/":
        base_folder += "/"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    params = {
        "TRAIN"     : f"{data_path}/train.txt",
        "VAL"       : f"{data_path}/valid.txt",
        "TEST"      : f"{data_path}/test.txt",
        "DIR"       : f"{data_path}",
    }

    LOAD_PARAM = {
        'batch_size' : 1,
        'shuffle'    : True,
        'pin_memory' : False,
        'num_workers': 4
    }

    with open(f'{data_path}/aaindex.json', 'r') as f:
        aaindex1 = json.load(f)

    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    train, valid, test = build_training_clusters(params, args.debug)

    train_set = PDBDataset(train, loader_pdb, params)
    train_loader = DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDBDataset(valid, loader_pdb, params)
    valid_loader = DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    model = SplitSeek(aaindex1=aaindex1,
                        node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_decoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        augment_eps=args.backbone_noise)

    model.to(device)

    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step']
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])

    else:
        total_step = 0
        epoch = 0

    # total_step = 0
    # epoch = 0

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
    min_valid_loss = 10

    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    pdb_dict_train = get_pdbs(train_loader, 1, args.max_protein_length, args.num_examples_per_epoch)
    pdb_dict_valid = get_pdbs(valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch)

    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)

    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)

    for e0 in range(args.num_epochs):
        t0 = time.time()
        e = epoch + e0
        model.train()
        train_sum, train_weight = 0., 0.

        for k, batch in enumerate(loader_train):
            X, S, ESM_S, L, mask, lengths, residue_idx, mask_self, chain_encoding_all, batch_name = featurize(batch, device)
            optimizer.zero_grad()
            mask_for_loss = mask*mask_self

            if args.mixed_precision:
                with torch.amp.autocast('cuda'):
                    probs = model(X, S, ESM_S, mask, residue_idx, chain_encoding_all)
                    loss, loss_av = loss_mse(L ,probs, mask_for_loss)

                scaler.scale(loss_av).backward()

                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                scaler.step(optimizer)
                scaler.update()

            else:
                probs = model(X, S, mask, residue_idx, chain_encoding_all)
                loss, loss_av = loss_mse(L ,probs, mask)

                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                optimizer.step()

            train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            train_weight += torch.sum(mask_for_loss).cpu().data.numpy()


            total_step += 1

        model.eval()
        with torch.no_grad():
            validation_sum, validation_weights = 0., 0.
            for _, batch in enumerate(loader_valid):
                X, S, ESM_S, L, mask, lengths, residue_idx, mask_self, chain_encoding_all, batch_name = featurize(batch, device)
                probs = model(X, S, ESM_S, mask, residue_idx, chain_encoding_all)
                mask_for_loss = mask*mask_self
                loss, loss_av = loss_mse(L, probs, mask)

                validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()

        train_loss = train_sum / (train_weight + 1e-9)
        train_perplexity = np.exp(1 + train_loss)
        validation_loss =  validation_sum / (validation_weights + 1e-9)
        validation_perplexity = np.exp(1 + validation_loss)

        train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)
        validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)

        t1 = time.time()
        dt = np.format_float_positional(np.float32(t1 - t0), unique=False, precision=1)
        with open(logfile, 'a') as f:
            f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}\n')

        print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}')

        checkpoint_filename_last = base_folder + 'model_weights/epoch_last.pt'.format(e+1, total_step)
        best_checkpoint_filename = base_folder + 'model_weights/best_weights.pt'
        torch.save({
                    'epoch': e+1,
                    'step': total_step,
                    'num_edges' : args.num_neighbors,
                    'noise_level': args.backbone_noise,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, checkpoint_filename_last)

        if validation_perplexity <= min_valid_loss:
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, best_checkpoint_filename)      

            min_valid_loss = validation_perplexity

        if (e+1) % args.save_model_every_n_epochs == 0:
            checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
            torch.save({
                    'epoch': e+1,
                    'step': total_step,
                    'num_edges' : args.num_neighbors,
                    'noise_level': args.backbone_noise, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, checkpoint_filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="./Fine_tuning_data", help="path for loading training data") 
    argparser.add_argument("--path_for_outputs", type=str, default="./finetune_outputs", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="../weights/pretrain.pt", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=50, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=2, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=10, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.1, help="amount of noise added to backbone during training")   
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
 
    args = argparser.parse_args()    
    main(args)   
