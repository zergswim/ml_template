# -*- coding: utf-8 -*-
import timm
import pandas as pd
import os
import numpy as np

DIR = '/opt/ml/input/data'
df = pd.read_csv(DIR+'/train/train.csv')

dfms = pd.DataFrame(columns=['gender', 'age', 'mask', 'img_path'])
arr = {'incorrect_mask':1, 'mask1':0, 'mask2':0, 'mask3':0, 'mask4':0, 'mask5':0, 'normal':2}

for key, val in arr.items():
    temp = pd.DataFrame(columns=['gender', 'age', 'mask', 'img_path'])
    map_gender = {'male':0, 'female':1}
    temp = df[['gender', 'age']].copy()
    temp.loc[:,('gender')] = df[['gender']].applymap(map_gender.get)
    temp.loc[:,('mask')] = val
    temp.loc[:,('img_path')] = DIR +'/train/images/'+df['path']
    temp.loc[:,('filename')] = key
    temp.loc[:,('age')] =   np.where(df['age']<30, 0,
                            np.where(df['age']<60, 1, 2))

    dfms = pd.concat([dfms, temp])

def GetExt(path, filename):
    with os.scandir(f'{path}/') as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                name, ext = entry.name.split('.')
                if filename == name: return ext    
    
for idx, path in enumerate(dfms['img_path']):
    filename = dfms['filename'].iloc[idx]
    dfms['filename'].iloc[idx] += '.' + GetExt(path, filename)

dfms['label'] = dfms['mask'] * 6 + dfms['gender'] * 3 + dfms['age']
dfms['img_path_full'] = dfms['img_path'] + '/' + dfms['filename']
dfms.head()
# print(dfms['mask'].value_counts())
# print(dfms['gender'].value_counts())
# print(dfms['age'].value_counts())

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import time
import gc
import pandas as pd
from PIL import Image
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
import random
from tqdm import tqdm

class ML():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transforms, dataloaders, dataset_sizes, class_names = None, None, None, None
    model, ciriterion, optimizer = None, None, None

    def __init__(self, transform, model, criterion, opt, lr):
        self.FixSeed(2368)
        self.Empty()
        self.model = model.to(self.device)
        self.criterion = criterion
#         self.optimizer = opt(model.fc.parameters(), lr=lr)
        self.optimizer = opt(model.parameters(), lr=lr) #, weight_decay=0.0)
        self.optimizer.zero_grad() #초기화
        self.transforms = transform
        
    def FixSeed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def Empty(self):
        torch.cuda.empty_cache()
        gc.collect()        
        
    def Data(self, train_sets, valid_sets, test_sets, batch_size=4, shuffle=True, num_workers=2):
#         train_sets, valid_sets = datasets.ImageFolder(train_dir, self.transforms['train']), datasets.ImageFolder(valid_dir, self.transforms['valid'])
        self.dataloaders = { 'train':DataLoader(train_sets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), 
                             'valid':DataLoader(valid_sets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
                             'test':DataLoader(test_sets, batch_size=batch_size, shuffle=False, num_workers=num_workers) }
        self.dataset_sizes = {'train':len(train_sets), 'valid':len(valid_sets), 'test':len(test_sets)}
        self.class_names = train_sets.classes
        print("train: {}, valid: {}, test: {}".format(self.dataset_sizes['train'], self.dataset_sizes['valid'], self.dataset_sizes['test']))
        
    def Train(self, num_epochs=5, save_stop=False, empty=True, name="Noname"):
        if empty: self.Empty()
        since = time.time()
        old_acc = 0.
#         with wandb.init(project="ClassTest_2"):
#             wandb.watch(self.model, self.criterion, log="all", log_freq=10)
            
        for epoch in range(num_epochs):
            print('[Epoch {}/{}]'.format(epoch+1, num_epochs))
            for phase in ['train', 'valid']:
                if phase == 'train': self.model.train()
                else: self.model.eval()

                with torch.set_grad_enabled(phase == 'train'):
                    running_loss, running_corrects = 0.0, 0
                    for inputs, labels in self.dataloaders[phase]:
#                     for inputs, labels in tqdm(self.dataloaders[phase]):
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        if phase == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                        _, preds = torch.max(outputs, 1)
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    epoch_loss, epoch_acc = running_loss / self.dataset_sizes[phase], running_corrects.double() / self.dataset_sizes[phase]
                    print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                    
#                 if save_stop and phase == 'valid':
#                     if old_acc > epoch_acc: print("No saved! (old_acc:{:.2f}, epoch_acc:{:.2f})".format(old_acc, epoch_acc)); break
#                     else: self.Save("{}_{:.2f}.ml".format(name, epoch_acc*100))
#                     old_acc = epoch_acc
                        
    #                     wandb.log({phase+"_loss": epoch_loss, phase+"_acc": epoch_acc}, step=epoch)

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
             
    def Save(self, filename):
        torch.save(self.model.state_dict(), filename)
        print(filename, 'saved')
        
    def Load(self, filename, empty=True):
        if empty: self.Empty()
        self.model.load_state_dict(torch.load(filename, map_location=self.device))        
        self.model.eval()
        print(filename, 'loaded')
        
    def Valid(self, show_yn=False):
        with torch.no_grad():
            running_corrects = 0
            for idx, (inputs, labels) in enumerate(self.dataloaders['valid']):
                outputs = self.model(inputs.to(self.device))
                probs = torch.nn.Softmax(dim=1)(outputs).cpu().detach().numpy()
                probs_max = np.max(probs, axis=1)
                preds_max = np.argmax(probs, axis=1)
                if show_yn and preds_max.all() != labels.numpy().all(): self.Show(inputs, preds_max, probs_max, labels)
                running_corrects += np.sum(preds_max == labels.numpy())
            print("acc {} valided".format(running_corrects / self.dataset_sizes['valid']))
        
    def Show(self, inputs, preds, vals, labels, figsize=(16,4)):
        plt.subplots(figsize=figsize)
        for idx, i in enumerate(inputs):
            plt.subplot(1, len(inputs), idx+1)
            plt.imshow(i.numpy().transpose((1, 2, 0)))
            plt.title("{} {} {:.2f}".format(preds[idx]==labels[idx], self.class_names[preds[idx]], vals[idx]))
            plt.axis('off')
        plt.show()

#     def TestSet(self, test_sets):
#         self.testloader = tqdm(DataLoader(test_sets, shuffle=False))
        
#     def Predict(self, test_sets):
    def Predict(self):
        with torch.no_grad():
#             test_loader = DataLoader(test_sets, shuffle=False, batch_size=128)
            answers = []
            for inputs, labels in tqdm(self.dataloaders['test']):
#             for inputs, labels in tqdm(test_loader):
                outputs = self.model(inputs.to(self.device))
                probs = torch.nn.Softmax(dim=1)(outputs).cpu().detach().numpy()
                answers.extend(probs.squeeze())
    #             answers.append(outputs.cpu().detach().numpy())

    #     # Save the model in the exchangeable ONNX format
    #     torch.onnx.export(model, inputs, "model.onnx")
    #     wandb.save("model.onnx")
        return answers  
              
    def TSNE(self):
        with torch.no_grad():
#             test_loader = DataLoader(test_sets, shuffle=False, batch_size=128)
            actual, deep_features = [], []
            for inputs, _ in tqdm(self.dataloaders['test']):
                outputs = self.model(inputs.to(self.device))
                deep_features += outputs.cpu().tolist()
                _, preds = torch.max(outputs, 1)
                actual += preds.cpu().tolist()
#                 actual += preds.squeeze().cpu().tolist()
                
            tsne = TSNE(n_components=2, random_state=0)
            cluster = np.array(tsne.fit_transform(np.array(deep_features)))
            actual = np.array(actual)
            
            plt.figure(figsize=(16,16))
            for i, label in enumerate(self.class_names):
                idx = np.where(actual==i)
                plt.scatter(cluster[idx,0], cluster[idx,1], marker='.', label=label)
            plt.legend()
            plt.axis('off')
            plt.legend()
            plt.show()
            plt.savefig('t-SNE_org.png')
    
#     def __str__(self):
#         return "dataloaders : {}".format(self.dataloaders)

transform = { 
    'train': transforms.Compose([ 
                        transforms.ToTensor(), 
#                         transforms.RandomVerticalFlip(),        
                        transforms.RandomHorizontalFlip(),        
#                         transforms.Grayscale(num_output_channels=3),
#                         transforms.RandomGrayscale(),
                        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
#                         transforms.RandomResizedCrop(224),
#                         transforms.Resize((224,224)),
#                         transforms.CenterCrop(224),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]),
    'valid': transforms.Compose([ 
                        transforms.ToTensor(),
#                         transforms.CenterCrop(224),
#                         transforms.Grayscale(num_output_channels=3),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]) }

class MyDataset(Dataset):
    def __init__(self, df, transform=None, classes=None, kind='all'):
        self.df, self.transform, self.classes, self.kind = df, transform, classes, kind
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
#         image = Image.open(data['img_path_full'])
#         print(data['img_path'])
        image = cv2.imread(data['img_path_full'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         if self.transform: image = self.transform(image=image)["image"]
        image = self.transform(image)
        if self.kind=='all': return image, data['label']
        else: return image, data[self.kind]
    def __len__(self): return len(self.df)

df_classes = ['wear-male-young','wear-male-middle','wear-male-old',
              'wear-female-young','wear-female-middle','wear-female-old',
              'incorr-male-young','incorr-male-middle','incorr-male-old',
              'incorr-female-young','incorr-female-middle','incorr-female-old',
              'notwear-male-young','notwear-male-middle','notwear-male-old',
              'notwear-female-young','notwear-female-middle','notwear-female-old']

model = timm.create_model('resnet50', pretrained=True, num_classes=len(df_classes))
# model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=len(df_classes))
# model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True, num_classes=len(df_classes))

k_fold_num = 10
batch_size = 64 #len(df_classes)
num_epochs = 5
lr = 0.00005

# Data imbalance
# class_num = [220, 165, 132, 74, 132, 64, 32, 59, 50, 81, 489, 62, 44, 65, 26, 204, 84, 30, 72, 69, 42, 121, 52, 61, 115, 21, 64, 76, 91, 101, 173, 34, 118, 303, 33, 220, 142, 97, 233, 59, 85, 73, 181, 137, 99, 120, 173, 60, 629, 44]
# class_weight = [1-(x/sum(class_num)) for x in class_num]
# tensor_weights = torch.FloatTensor(class_weight).to(device='cuda:0')

df = dfms[['mask','gender','age','img_path_full', 'label']].copy()
X = dfms[['mask','gender','age','img_path_full']].values.tolist()
y = dfms['label'].values.tolist()

# Train for kfold
kfold = StratifiedKFold(n_splits=k_fold_num)

for i, (train_index, valid_index) in enumerate(kfold.split(X, y)):
    print(f"\n** Fold_{i} **")
    train_sets = MyDataset(df.iloc[train_index], transform['train'], df_classes, 'all')
    valid_sets = MyDataset(df.iloc[valid_index], transform['valid'], df_classes, 'all')
    ml = ML(transform, model, torch.nn.CrossEntropyLoss(), torch.optim.Adam, lr=lr)
#     ml = ML(transform, model, torch.nn.CrossEntropyLoss(weight=tensor_weights), torch.optim.Adam, lr=lr)
    ml.Data(train_sets, valid_sets, valid_sets, batch_size=batch_size)
#     ml.Load('/kaggle/input/mldata/Fold_0_91.72.ml')
    ml.Train(num_epochs=num_epochs, name=f"Fold_{i}", save_stop=True)
#     ml.TSNE(valid_sets, prediction=True)
    ml.Save("train_fold_{}.ml".format(i))

df_test = pd.read_csv(DIR+'/eval/info.csv')
df_test['label'] = 0
df_test['img_path_full'] = DIR+'/eval/images/' +  df_test['ImageID']

len_img, len_classes = len(df_test), len(df_classes)
print(len_img, len_classes)

# ML_DIR = "/kaggle/input/mldata/"
ML_DIR = ""
test_sets = MyDataset(df_test, transform['valid'], df_classes)

rst = [[np.zeros(len_classes)] for i in range(len_img)]

#ensemble 처리
# ml = ML(transform, model, torch.nn.CrossEntropyLoss(weight=tensor_weights), torch.optim.Adam, lr=lr)
ml = ML(transform, model, torch.nn.CrossEntropyLoss(), torch.optim.Adam, lr=lr)
ml.Data(test_sets, test_sets, test_sets, batch_size=128)

for i in range(k_fold_num):    
#     ml.Load(ML_DIR + "Fold_{}_{:.2f}.ml".format(i,fold_score[i]))
    ml.Load(ML_DIR + "train_fold_{}.ml".format(i))
#     ml.Valid(show_yn=True)
    rtn = ml.Predict()
#     print("rtn:",len(rtn))
    for idx, r in enumerate(rtn):
        rst[idx] += r #probs sum
#         r = r.flatten()
#         rst[idx] += np.array([1 if max(r)==i else 0 for i in r]) #voting

# for idx, r in enumerate(rst):
#     print(idx, type(r), len(r))

answers = [np.argmax(r/len_classes) for r in rst]

submit = pd.DataFrame()
submit["ImageID"] = df_test["ImageID"]
submit["ans"] = answers
# submit["ans"] = submission_df["ans"]
submit.to_csv('kflod10_resnet50_base_epoch5.csv', index=False)

submit['ans'].value_counts()

#TSNE 분석
ml.TSNE()

#plt.plot([1,2,3,4])

