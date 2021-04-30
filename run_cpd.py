import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--models-dir', metavar='PATH', default='./models/',
                    help='directory to save trained models, default=./models/')
parser.add_argument('--num-workers', metavar='N', type=int, default=4,
                   help='number of threads for loading data, default=4')
parser.add_argument('--max-nodes', metavar='N', type=int, default=3000,
                    help='max number of nodes per batch, default=3000')
parser.add_argument('--epochs', metavar='N', type=int, default=100,
                    help='training epochs, default=100')
parser.add_argument('--cath-data', metavar='PATH', default='./data/chain_set.jsonl',
                    help='location of CATH dataset, default=./data/chain_set.jsonl')
parser.add_argument('--cath-splits', metavar='PATH', default='./data/chain_set_splits.json',
                    help='location of CATH split file, default=./data/chain_set_splits.json')
parser.add_argument('--ts50', metavar='PATH', default='./data/ts50.json',
                    help='location of TS50 dataset, default=./data/ts50.json')
parser.add_argument('--train', action="store_true", help="train a model")
parser.add_argument('--test-r', metavar='PATH', default=None,
                    help='evaluate a trained model on recovery (without training)')
parser.add_argument('--test-p', metavar='PATH', default=None,
                    help='evaluate a trained model on perplexity (without training)')
parser.add_argument('--n-samples', metavar='N', default=100,
                    help='number of sequences to sample (if testing recovery), default=100')

args = parser.parse_args()
assert sum(map(bool, [args.train, args.test_p, args.test_r])) == 1, \
    "Specify exactly one of --train, --test_r, --test_p"

import torch
import torch.nn as nn
import gvp.data, gvp.models
from datetime import datetime
import tqdm, os, json
import numpy as np
from sklearn.metrics import confusion_matrix
import torch_geometric
from functools import partial
print = partial(print, flush=True)

node_dim = (100, 16)
edge_dim = (32, 1)
device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(args.models_dir): os.makedirs(args.models_dir)
model_id = int(datetime.timestamp(datetime.now()))
dataloader = lambda x: torch_geometric.data.DataLoader(x, 
                        num_workers=args.num_workers,
                        batch_sampler=gvp.data.BatchSampler(
                            x.node_counts, max_nodes=args.max_nodes))

def main():
    
    model = gvp.models.CPDModel((6, 3), node_dim, (32, 1), edge_dim).to(device)
    
    print("Loading CATH dataset")
    cath = gvp.data.CATHDataset(path="data/chain_set.jsonl",
                                splits_path="data/chain_set_splits.json")    
    
    trainset, valset, testset = map(gvp.data.ProteinGraphDataset,
                                    (cath.train, cath.val, cath.test))
    
    if args.test_r or args.test_p:
        ts50set = gvp.data.ProteinGraphDataset(json.load(open(args.ts50)))
        model.load_state_dict(torch.load(args.test_r or args.test_p))
    
    if args.test_r:
        print("Testing on CATH testset"); test_recovery(model, testset)
        print("Testing on TS50 set"); test_recovery(model, ts50set)
    
    elif args.test_p:
        print("Testing on CATH testset"); test_perplexity(model, testset)
        print("Testing on TS50 set"); test_perplexity(model, ts50set)
    
    elif args.train:
        train(model, trainset, valset, testset)
    
    
def train(model, trainset, valset, testset):
    train_loader, val_loader, test_loader = map(dataloader,
                    (trainset, valset, testset))
    optimizer = torch.optim.Adam(model.parameters())
    best_path, best_val = None, np.inf
    lookup = train_loader.dataset.num_to_letter
    for epoch in range(args.epochs):
        model.train()
        loss, acc, confusion = loop(model, train_loader, optimizer=optimizer)
        path = f"{args.models_dir}/{model_id}_{epoch}.pt"
        torch.save(model.state_dict(), path)
        print(f'EPOCH {epoch} TRAIN loss: {loss:.4f} acc: {acc:.4f}')
        print_confusion(confusion, lookup=lookup)
        
        model.eval()
        with torch.no_grad():
            loss, acc, confusion = loop(model, val_loader)    
        print(f'EPOCH {epoch} VAL loss: {loss:.4f} acc: {acc:.4f}')
        print_confusion(confusion, lookup=lookup)
        
        if loss < best_val:
            best_path, best_val = path, loss
        print(f'BEST {best_path} VAL loss: {best_val:.4f}')
        
    print(f"TESTING: loading from {best_path}")
    model.load_state_dict(torch.load(best_path))
    
    model.eval()
    with torch.no_grad():
        loss, acc, confusion = loop(model, test_loader)
    print(f'TEST loss: {loss:.4f} acc: {acc:.4f}')
    print_confusion(confusion,lookup=lookup)

def test_perplexity(model, dataset):
    model.eval()
    with torch.no_grad():
        loss, acc, confusion = loop(model, dataloader(dataset))
    print(f'TEST perplexity: {np.exp(loss):.4f}')
    print_confusion(confusion, lookup=dataset.num_to_letter)

def test_recovery(model, dataset):
    recovery = []
    
    for protein in tqdm.tqdm(dataset):
        protein = protein.to(device)
        h_V = (protein.node_s, protein.node_v)
        h_E = (protein.edge_s, protein.edge_v) 
        sample = model.sample(h_V, protein.edge_index, 
                              h_E, n_samples=args.n_samples)
        
        recovery_ = sample.eq(protein.seq).float().mean().cpu().numpy()
        recovery.append(recovery_)
        print(protein.name, recovery_, flush=True)

    recovery = np.median(recovery)
    print(f'TEST recovery: {recovery:.4f}')
    
def loop(model, dataloader, optimizer=None):

    confusion = np.zeros((20, 20))
    t = tqdm.tqdm(dataloader)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total_count = 0, 0, 0
    
    for batch in t:
        if optimizer: optimizer.zero_grad()
    
        batch = batch.to(device)
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        
        logits = model(h_V, batch.edge_index, h_E, seq=batch.seq)
        logits, seq = logits[batch.mask], batch.seq[batch.mask]
        loss_value = loss_fn(logits, seq)

        if optimizer:
            loss_value.backward()
            optimizer.step()

        num_nodes = int(batch.mask.sum())
        total_loss += float(loss_value) * num_nodes
        total_count += num_nodes
        pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        true = seq.detach().cpu().numpy()
        total_correct += (pred == true).sum()
        confusion += confusion_matrix(true, pred, labels=range(20))
        t.set_description("%.5f" % float(total_loss/total_count))
        
        torch.cuda.empty_cache()
        
    return total_loss / total_count, total_correct / total_count, confusion
    
def print_confusion(mat, lookup):
    counts = mat.astype(np.int32)
    mat = (counts.T / counts.sum(axis=-1, keepdims=True).T).T
    mat = np.round(mat * 1000).astype(np.int32)
    res = '\n'
    for i in range(20):
        res += '\t{}'.format(lookup[i])
    res += '\tCount\n'
    for i in range(20):
        res += '{}\t'.format(lookup[i])
        res += '\t'.join('{}'.format(n) for n in mat[i])
        res += '\t{}\n'.format(sum(counts[i]))
    print(res)
    
if __name__== "__main__":
    main()