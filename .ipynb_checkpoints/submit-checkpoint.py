from base import ML
import pre
import torch
import pandas as pd
import numpy as np

# dfms = pre.Preprocessing()
df_classes = pre.Df_classes
trfm = pre.Transform
MyDataset = pre.MyDataset
model = pre.Model

k_fold_num = 10

DIR = '/opt/ml/input/data'
df_test = pd.read_csv(DIR+'/eval/info.csv')
df_test['label'] = 0
df_test['img_path_full'] = DIR+'/eval/images/' +  df_test['ImageID']

len_img, len_classes = len(df_test), len(df_classes)
print(len_img, len_classes)

# ML_DIR = "/kaggle/input/mldata/"
ML_DIR = ""
test_sets = MyDataset(df_test, trfm['valid'], df_classes)

rst = [[np.zeros(len_classes)] for i in range(len_img)]

#ensemble 처리
# ml = ML(transform, model, torch.nn.CrossEntropyLoss(weight=tensor_weights), torch.optim.Adam, lr=lr)
ml = ML(trfm, model, torch.nn.CrossEntropyLoss(), torch.optim.Adam, lr=0)
ml.Data(test_sets, test_sets, test_sets, batch_size=128)

for i in range(k_fold_num):    
#     ml.Load(ML_DIR + "Fold_{}_{:.2f}.ml".format(i,fold_score[i]))
    ml.Load(ML_DIR + "train_fold_{}.ml".format(i))
#     ml.Valid(show_yn=True)
    rtn = ml.Predict()
    temp = np.array(rtn)
    df_temp = pd.DataFrame(temp)
    df_temp.to_csv('train_fold_{}.csv'.format(i), index=False)

#     print("rtn:",len(rtn))
    for idx, r in enumerate(rtn):
        rst[idx] += r #probs sum
#         r = r.flatten()
#         rst[idx] += np.array([1 if max(r)==i else 0 for i in r]) #voting

answers = [np.argmax(r/len_classes) for r in rst]

submit = pd.DataFrame()
submit["ImageID"] = df_test["ImageID"]
submit["ans"] = answers
submit.to_csv('kflod10.csv', index=False)

submit['ans'].value_counts()

#TSNE 분석
ml.TSNE()

#plt.plot([1,2,3,4])

