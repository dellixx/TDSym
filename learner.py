import argparse
from typing import Dict
import torch
from torch import optim
from datasets import TemporalDataset
from optimizers import TKBCOptimizer

from models import get_model_class
from regularizers import N3, Spiral3
import os
import datetime

import random
import os
import numpy as np
import torch

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🔒 Random seed fixed to {seed}")


def count_parameters(model):
    table = []
    total_params = 0
    print(f"\n{'LAYER':<30} {'SHAPE':<25} {'PARAMS':<15}")
    print("="*75)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.append(f"{name:<30} {str(list(parameter.shape)):<25} {params:<15,}")
        total_params += params
    
    print("\n".join(table))
    print("="*75)
    print(f"Total Trainable Params: {total_params:,}")
    print(f"Theoretical Model Size: {total_params * 4 / (1024 * 1024):.2f} MB")
    print("="*75 + "\n")
    return total_params


parser = argparse.ArgumentParser(description="Train Temporal Knowledge Graph Embedding Models")
parser.add_argument('--dataset', type=str, default='ICEWS14', help="Dataset name")
parser.add_argument('--model', default='TDSym', type=str, help="Model Name") 
parser.add_argument('--max_epochs', default=100, type=int, help="Number of epochs.")
parser.add_argument('--valid_freq', default=5, type=int, help="Number of epochs between each valid.")
parser.add_argument('--rank', default=32, type=int, help="Factorization rank.")
parser.add_argument('--batch_size', default=1000, type=int, help="Batch size.")
parser.add_argument('--learning_rate', default=0.1, type=float, help="Learning rate")
parser.add_argument('--emb_reg', default=0., type=float, help="Embedding regularizer strength")
parser.add_argument('--time_reg', default=0., type=float, help="Timestamp regularizer strength")
parser.add_argument('--no_time_emb', default=False, action="store_true", help="Use a specific embedding for non temporal relations")

args = parser.parse_args()

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}

def learn(model=args.model,
          dataset=args.dataset,
          rank=args.rank,
          learning_rate = args.learning_rate,
          batch_size = args.batch_size, 
          emb_reg=args.emb_reg, 
          time_reg=args.time_reg
          ):
    set_seed()
    root = 'results/'+ dataset +'/' + model
    model_name = model
    datasetname = dataset

    PATH=os.path.join(root,f'rank{rank:.0f}/lr{learning_rate:.4f}/batch{batch_size:.0f}/emb_reg{emb_reg:.5f}/time_reg{time_reg:.5f}/')
    
    dataset = TemporalDataset(dataset)
    sizes = dataset.get_shape()
    
    print(f"Initializing Model: {model_name}...")
    ModelClass = get_model_class(model_name)
    model = ModelClass(sizes, rank)
    
    model = model.cuda()

    print("\n🚀 [Model Configuration]")
    count_parameters(model)

    opt = optim.Adagrad(model.parameters(), lr=learning_rate)

    print("Start training process: ", model_name, "on", datasetname, "using", "rank =", rank, "lr =", learning_rate, "emb_reg =", emb_reg, "time_reg =", time_reg)

    emb_reg = N3(emb_reg)
    time_reg = Spiral3(time_reg)
  
    try:
        os.makedirs(PATH)
    except FileExistsError:
        pass
    patience = 0
    mrr_std = 0

    curve = {'train': [], 'valid': [], 'test': []}

    for epoch in range(args.max_epochs):
        print("[ Epoch:", epoch, "]")
        examples = torch.from_numpy(
            dataset.get_train().astype('int64')
        )

        model.train()

        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, opt,
            batch_size=batch_size
        )

        optimizer.epoch(examples)
       
        if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
            if dataset.interval: 
                valid, test = [
                    avg_both(*dataset.eval(model, split, -1))
                    for split in ['valid', 'test']
                ]
                print("valid: ", valid['MRR'])
                print("test: ", test['MRR'])
            else:
                valid, test, train = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test', 'train']
                ]
                print("valid: ", valid['MRR'])
                print("test: ", test['MRR'])
                print("train: ", train['MRR'])

            # Save results
            f = open(os.path.join(PATH, 'result.txt'), 'a+')
            f.write("\n[Epoch:{}]-VALID : ".format(epoch))
            f.write(str(valid))
            f.close()
            
            mrr_valid = valid['MRR']
            if mrr_valid < mrr_std:
               patience += 1
               if patience >= 5:
                  print("Early stopping ...")
                  break
            else:
               patience = 0
               mrr_std = mrr_valid
               torch.save(model.state_dict(), os.path.join(PATH, model_name+'.pkl'))

            curve['valid'].append(valid)
            if not dataset.interval:
                curve['train'].append(train)
    
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"\t TRAIN:-{current_time} ", train)
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\t VALID -{current_time}: ", valid)
            print(f"\t TEST -{current_time}: ", test)

    model.load_state_dict(torch.load(os.path.join(PATH, model_name+'.pkl')))
    results = avg_both(*dataset.eval(model, 'test', -1))
    print("\n\nTEST : ", results)
    f = open(os.path.join(PATH, 'result.txt'), 'a+')
    f.write("\n\nTEST : ")
    f.write(str(results))
    f.close()

if __name__ == '__main__':
    learn()