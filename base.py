# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import gc

from sklearn.manifold import TSNE
import random
from tqdm import tqdm
import numpy as np

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
