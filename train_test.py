import os
import time
import numpy as np
import random
import copy
import scipy.io as sio
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow import Tensor
from transformers import get_cosine_schedule_with_warmup

from S2CAT import S2CAT
from utils.FocalLoss import FocalLoss
from utils.CreateCube import create_patch
from utils.SplitDataset import split_dataset


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='PU',help='dataset')
parser.add_argument('--train_ratio', default=0.007,help='ratio of train')
parser.add_argument('--epochs',default=300,help='epoch')
parser.add_argument('--lr',default=0.001,help='learning rate')
parser.add_argument('--gamma',default=0.4,help='gamma of focal loss')
args = parser.parse_args()

config={'batch_size':64,
        'dataset':args.dataset,
        'model':'S2CAT',
        'train_ratio':float(args.train_ratio),
        'weight_decay':0.001,
        'epoch':int(args.epochs),
        'patch_size':9,
        'fc_dim':16,
        'heads':2,
        'drop':0.1,
        'lr':float(args.lr),
        'gamma':float(args.gamma),
        }

def load_data(dataset):
    if dataset == 'PU': # 0.7% 300epoch  9class
        data = sio.loadmat('./data/Pavia_university/PaviaU.mat')['paviaU']
        labels = sio.loadmat('./data/Pavia_university/PaviaU_gt.mat')['paviaU_gt']
    if dataset == 'IP': # 5% 300epoch  16class
        data = sio.loadmat('./data/Indian_pines/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat('./data/Indian_pines/Indian_pines_gt.mat')['indian_pines_gt']
    if dataset == 'BS': # 5% 300epoch
        data = sio.loadmat('./data/Botswana/Botswana.mat')['Botswana']
        labels = sio.loadmat('./data/Botswana/Botswana_gt.mat')['Botswana_gt']
    if dataset == 'SA': # 5% 100epoch
        data = sio.loadmat('./data/Salinas/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat('./data/Salinas/Salinas_gt.mat')['salinas_gt']
    if dataset == 'LK': # 0.1% 300epoch
        data = sio.loadmat('./data/LongKou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
        labels = sio.loadmat('./data/LongKou/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
    if dataset == 'HS': # 5% 500epoch  15class
        data = sio.loadmat('data/Houston/Houston.mat')['HSI']
        labels = sio.loadmat('data/Houston/Houston_gt.mat')['gt']
    if dataset == 'TR': # 1% 100epoch
        data = sio.loadmat('data/Trento/HSI_Trento.mat')['HSI_Trento']
        labels = sio.loadmat('data/Trento/GT_Trento.mat')['GT_Trento']
    if dataset == 'HR': # 3% 300epoch
        data = sio.loadmat('data/HR-L/Loukia.mat')['imggt']
        labels = sio.loadmat('data/HR-L/Loukia_gt.mat')['imggt']
    if dataset == 'HY': # 1% 200epoch
        data = sio.loadmat('data/HyRank/Dioni.mat')['ori_data']
        labels = sio.loadmat('data/HyRank/Dioni_gt_out68.mat')['map']
    if dataset == 'HH': # 1% 200epoch
        data = sio.loadmat('data/HongHu/WHU_Hi_HongHu.mat')['WHU_Hi_HongHu']
        labels = sio.loadmat('data/HongHu/WHU_Hi_HongHu_gt.mat')['WHU_Hi_HongHu_gt']
    if dataset == 'HC': # 1% 300epoch
        data = sio.loadmat('data/HanChuan/WHU_Hi_HanChuan.mat')['WHU_Hi_HanChuan']
        labels = sio.loadmat('data/HanChuan/WHU_Hi_HanChuan_gt.mat')['WHU_Hi_HanChuan_gt']
    if dataset == 'XZ': # 1% 300epoch
        data = sio.loadmat('data/XuZhou/data.mat')['all_y']
        labels = sio.loadmat('data/XuZhou/gt.mat')['all_y']
    else:
        print('Please add your dataset.')
    return data, labels

def random_num(size, end):
    range_ls = [i for i in range(end)]
    num_ls = []
    for i in range(size):
        num = random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls

def train(model,classes,config,device,sampler):
    epochs =  config['epoch']
    lr = config['lr']
    wd = config['weight_decay']
    dataset = config['dataset']
    gamma = config['gamma']
    model_name = config['model']
    model = model.to(device)
    criterion = FocalLoss(class_num=classes,gamma=gamma,alpha=sampler,use_alpha=True)
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 0.1*epochs*len(data_loader['train']), num_training_steps = epochs*len(data_loader['train']))
    best_acc = 0
    best_model = None
    best_epoch = 0
    start_time = time.time()
    for epoch in range(epochs):
        for phase in ['train','val']:
            running_loss = 0
            running_correct = 0
            for i, (data, target) in enumerate(data_loader[phase]):
                data, target = data.to(device), target.to(device)
                if phase == 'train':
                    model.train()
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                else:
                    model.eval()
                    with torch.no_grad():
                        outputs = model(data)
                        loss = criterion(outputs, target)
                running_loss += loss.item()*data.shape[0]
                _,outputs = torch.max(outputs,1)
                running_correct += torch.sum(outputs == target.data)
            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = running_correct.double() / len(data_loader[phase].dataset)
            
            if phase == 'val' and best_acc < epoch_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                
            print('[Epoch: %d] [%s] [loss: %.4f] [acc: %.4f]' % (epoch + 1,phase,epoch_loss,epoch_acc))
    end_time = time.time()
    training_time = round(end_time - start_time,4)
    print('Training time: %.4f' % training_time)
                
    path = f'./weights/{dataset}_{model_name}_{best_epoch}_{training_time}_mine.pth'
    torch.save(best_model.state_dict(),path)
    print('Finished Training')


    return best_epoch, training_time


def test(device, model, test_loader):
    count = 0
    model.eval()
    y_pred_test = 0
    y_test = 0
    start_time = time.time()
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))
    end_time = time.time()
    training_time = round(end_time - start_time, 4)
    print('Testing time: %.4f' % training_time)
    return y_pred_test, y_test
    
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

class to_dataset(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.len = x.shape[0]
        self.data = torch.tensor(x)    #torch.FloatTensor(x)
        self.gt = torch.LongTensor(y)
    def __getitem__(self,index):
        return self.data[index],self.gt[index]
    def __len__(self):
        return self.len

dataset_name = config['dataset']
model_name = config['model']
train_ratio = float(config['train_ratio'])
patch = config['patch_size']
data,gt = load_data(dataset_name)
batch_size=config['batch_size']


all_data,all_gt = create_patch(data,gt,patch_size = patch)
band = all_data.shape[-1]
classes = int(np.max(np.unique(all_gt)))
x_train,y_train,x_val,y_val,x_test,y_test,sampler = split_dataset(all_data,all_gt,train_ratio=train_ratio)
train_ds = to_dataset(x_train,y_train)
val_ds = to_dataset(x_val,y_val)
test_ds = to_dataset(x_test,y_test)
all_ds = to_dataset(all_data,all_gt)
train_loader = torch.utils.data.DataLoader(train_ds,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0)
                                           
val_loader = torch.utils.data.DataLoader(val_ds,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=0)
data_loader = {'train':train_loader,
               'val':val_loader}

test_loader = torch.utils.data.DataLoader(test_ds,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=0)
all_loader = torch.utils.data.DataLoader(all_ds,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=0)

model = S2CAT(channels=16,
            patch=config['patch_size'],
            bands=band,
            num_class=classes,
            fc_dim=config['fc_dim'],
            heads=config['heads'],
            drop=config['drop'],
            )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
best_epoch, training_time = train(model,classes,config,device,sampler)

path = f'./weights/{dataset_name}_{model_name}_{best_epoch}_{training_time}.pth'
model.load_state_dict(torch.load(path))
model = model.to(device)
p_t,t_t=test(device,model,test_loader)
OA, AA_mean, Kappa, AA=output_metric(t_t, p_t)
print('Test \n','OA: ',OA,'AA_mean: ',AA_mean,' Kappa: ',Kappa,'AA: ',AA)

# get_cls_map.get_cls_map(model, device, all_loader, gt)
