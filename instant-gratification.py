#import liblaries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import KernelPCA
import tensorflow as tf
from tensorflow import keras

#define functions
def score(y_true,y_pred):
    y_true[y_true==1] = 1 - 10**(-15)
    y_true[y_true==0] = 10**(-15)
    y_pred[y_pred==1] = 1 - 10**(-15)
    y_pred[y_pred==0] = 10**(-15)
    n = y_true.shape[0]
    cols = y_true.shape[1]
    if cols==1:
        o = np.ones(n).reshape(-1,1)
    else:
        o = np.ones((n,cols))

    s = y_true*np.log(y_pred) + (o - y_true)*np.log(o - y_pred)
    return(-s.sum().sum()/n/cols)

#read files
df_train = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")
df_test = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
df_y = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

#create features
for i in df_train['cp_time'].unique():
    for j in df_train['cp_dose'].unique():
        means = pd.concat([df_train.loc[(df_train['cp_type']=='ctl_vehicle')&(df_train['cp_time']==i)&(df_train['cp_dose']==j),'g-0':'c-99'],
                           df_test.loc[(df_test['cp_type']=='ctl_vehicle')&(df_test['cp_time']==i)&(df_test['cp_dose']==j),'g-0':'c-99']]).mean()
        stds = pd.concat([df_train.loc[(df_train['cp_type']=='ctl_vehicle')&(df_train['cp_time']==i)&(df_train['cp_dose']==j),'g-0':'c-99'],
                          df_test.loc[(df_test['cp_type']=='ctl_vehicle')&(df_test['cp_time']==i)&(df_test['cp_dose']==j),'g-0':'c-99']]).std()
        df_train.loc[(df_train['cp_time']==i)&(df_train['cp_dose']==j),'g-0':'c-99'] = (df_train.loc[(df_train['cp_time']==i)&(df_train['cp_dose']==j),'g-0':'c-99'] - means) / stds
        df_test.loc[(df_test['cp_time']==i)&(df_test['cp_dose']==j),'g-0':'c-99'] = (df_test.loc[(df_test['cp_time']==i)&(df_test['cp_dose']==j),'g-0':'c-99'] - means) / stds

#NN model
tf.random.set_seed(123)
model = keras.models.Sequential()
model.add(keras.layers.Dense(2000, input_shape=(872,),activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(206, activation='sigmoid'))
model.compile(optimizer='Adam',loss=keras.losses.BinaryCrossentropy(label_smoothing=0.001))

x_ind = df_train['sig_id']
x = df_train.iloc[:,4:]
y = df_y.iloc[:,1:]
test_ind = df_test['sig_id']
x_test = df_test.iloc[:,4:]
pred_val = pd.DataFrame(np.zeros(y.shape),columns=y.columns)
pred_test = pd.DataFrame(np.zeros((x_test.shape[0],y.shape[1])),columns=y.columns)

scl = RobustScaler()
x = scl.fit_transform(x)
x_test = scl.fit_transform(x_test)
#kpca = KernelPCA(n_components=200,kernel='cosine',n_jobs=-1).fit(x)
#x = kpca.transform(x)
#x_test = kpca.transform(x_test)
x = pd.DataFrame(x)
x_test = pd.DataFrame(x_test)

kf = StratifiedKFold(n_splits=10,random_state=123)

y_cols = y.columns
y['label'] = ''
for c in y_cols:
    y['label'] = y['label'].str.cat(y[c].astype('str'),sep='')
y_label = LabelEncoder().fit_transform(y['label'])
del y['label']
for train_ind,valid_ind in kf.split(x,y_label):
    x_train,x_valid = x.loc[train_ind,:],x.loc[valid_ind,:]
    y_train,y_valid = y.loc[train_ind,:],y.loc[valid_ind,:]
    
    reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')
    model.fit(np.array(x_train),np.array(y_train),epochs=30,validation_data=(np.array(x_valid),np.array(y_valid)),callbacks=[reduce_lr_loss])
    pred_val.loc[valid_ind,:] = model.predict(x_valid)
    pred_test.loc[:,:] += model.predict(x_test) / kf.get_n_splits()

print('valid score multi:',score(y,pred_val))

#submit files
pred_test.index = test_ind
submission = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")
submission.index = submission['sig_id']
submission.iloc[:,1:] = pred_test
submission = submission.reset_index(drop=True)
submission.to_csv("submission.csv",index=False)
