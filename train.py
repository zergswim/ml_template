from base import ML
import pre
import torch
from sklearn.model_selection import StratifiedKFold

dfms = pre.Preprocessing()
df_classes = pre.Df_classes
trfm = pre.Transform
MyDataset = pre.MyDataset
model = pre.Model

k_fold_num = 10
batch_size = 32 #len(df_classes)
num_epochs = 10
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
    train_sets = MyDataset(df.iloc[train_index], trfm['train'], df_classes, 'all')
    valid_sets = MyDataset(df.iloc[valid_index], trfm['valid'], df_classes, 'all')
    ml = ML(trfm, model, torch.nn.CrossEntropyLoss(), torch.optim.Adam, lr=lr)
#     ml = ML(transform, model, torch.nn.CrossEntropyLoss(weight=tensor_weights), torch.optim.Adam, lr=lr)
    ml.Data(train_sets, valid_sets, valid_sets, batch_size=batch_size)
#     ml.Load('/kaggle/input/mldata/Fold_0_91.72.ml')
    ml.Train(num_epochs=num_epochs, name=f"Fold_{i}", save_stop=True)
#     ml.TSNE(valid_sets, prediction=True)
    ml.Save("train_fold_{}.ml".format(i))
