import pandas as pd
import os
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2
from sklearn.preprocessing import LabelEncoder
import timm

def Preprocessing():
    DIR = '/opt/ml/ML/art/data'
    df = pd.read_csv(DIR+'/train.csv')

    df.loc[3896, ('artist')] = 'Titian'
    df.loc[3986, ('artist')] = 'Alfred Sisley'    

    dfms = pd.DataFrame(columns=['img_path', 'label'])

    dfms['img_path'] = df['img_path'].str.replace('./train/', DIR+'/train/', regex=False)

    le = LabelEncoder()
    dfms['label'] = le.fit_transform(df['artist'])
    # df_classes = le.classes_

    # arr = {'incorrect_mask':1, 'mask1':0, 'mask2':0, 'mask3':0, 'mask4':0, 'mask5':0, 'normal':2}

    # for key, val in arr.items():
    #     temp = pd.DataFrame(columns=['gender', 'age', 'mask', 'img_path'])
    #     map_gender = {'male':0, 'female':1}
    #     temp = df[['gender', 'age']].copy()
    #     temp.loc[:,('gender')] = df[['gender']].applymap(map_gender.get)
    #     temp.loc[:,('mask')] = val
    #     temp.loc[:,('img_path')] = DIR +'/train/images/'+df['path']
    #     temp.loc[:,('filename')] = key
    #     temp.loc[:,('age')] =   np.where(df['age']<30, 0,
    #                             np.where(df['age']<60, 1, 2))

    #     dfms = pd.concat([dfms, temp])

    # def GetExt(path, filename):
    #     with os.scandir(f'{path}/') as it:
    #         for entry in it:
    #             if not entry.name.startswith('.') and entry.is_file():
    #                 name, ext = entry.name.split('.')
    #                 if filename == name: return ext    
        
    # for idx, path in enumerate(dfms['img_path']):
    #     filename = dfms['filename'].iloc[idx]
    #     dfms['filename'].iloc[idx] += '.' + GetExt(path, filename)

    # dfms['label'] = dfms['mask'] * 6 + dfms['gender'] * 3 + dfms['age']
    # dfms['img_path_full'] = dfms['img_path'] + '/' + dfms['filename']
    # print(dfms.head())
    # print(dfms['mask'].value_counts())
    # print(dfms['gender'].value_counts())
    # print(dfms['age'].value_counts())
    return dfms

class MyDataset(Dataset):
    def __init__(self, df, transform=None, classes=None):
        self.df, self.transform, self.classes = df, transform, classes
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
#         image = Image.open(data['img_path'])
#         print(data['img_path'])
        image = cv2.imread(data['img_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform: image = self.transform(image=image)["image"]
#        if self.transform: image = self.transform(image)
        return image, data['label']
    def __len__(self): return len(self.df)

Df_classes = ['Albrecht Du rer', 'Alfred Sisley', 'Amedeo Modigliani', 'Andrei Rublev',
 'Andy Warhol', 'Camille Pissarro', 'Caravaggio', 'Claude Monet',
 'Diego Rivera', 'Diego Velazquez', 'Edgar Degas', 'Edouard Manet',
 'Edvard Munch', 'El Greco', 'Eugene Delacroix', 'Francisco Goya',
 'Frida Kahlo', 'Georges Seurat', 'Giotto di Bondone', 'Gustav Klimt',
 'Gustave Courbet', 'Henri Matisse', 'Henri Rousseau',
 'Henri de Toulouse-Lautrec', 'Hieronymus Bosch', 'Jackson Pollock',
 'Jan van Eyck', 'Joan Miro', 'Kazimir Malevich', 'Leonardo da Vinci',
 'Marc Chagall', 'Michelangelo', 'Mikhail Vrubel', 'Pablo Picasso',
 'Paul Cezanne', 'Paul Gauguin', 'Paul Klee', 'Peter Paul Rubens',
 'Pierre-Auguste Renoir', 'Piet Mondrian', 'Pieter Bruegel', 'Raphael',
 'Rembrandt', 'Rene Magritte', 'Salvador Dali', 'Sandro Botticelli', 'Titian',
 'Vasiliy Kandinskiy', 'Vincent van Gogh', 'William Turner']

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

Transform = { 
    'train': A.Compose([ 
                        A.Resize(224*2,224*2),
                        A.RandomCrop(width=224, height=224),
                        A.HorizontalFlip(p=0.5),
                        # A.RandomBrightnessContrast(p=0.2),        
                        A.VerticalFlip(p=0.5),              
                        A.RandomRotate90(p=0.5),        
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                        ToTensorV2()
#                         transforms.ToTensor(), 
#                         transforms.RandomVerticalFlip(),        
#                         transforms.RandomHorizontalFlip(),        
#                         transforms.Grayscale(num_output_channels=3),
#                         transforms.RandomGrayscale(),
#                         transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
#                         transforms.RandomResizedCrop(224),
#                         transforms.Resize((224,224)),
#                         transforms.CenterCrop(224),
#                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]),
    'valid': A.Compose([ 
                        A.Resize(224,224),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                        ToTensorV2()
#                         transforms.ToTensor(),
#                         transforms.Resize((224,224)),
#                         transforms.CenterCrop(224),
#                         transforms.Grayscale(num_output_channels=3),
#                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]) }

# Model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=len(Df_classes))
# Model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=len(Df_classes))
Model = timm.create_model('resnet50', pretrained=True, num_classes=len(Df_classes))

