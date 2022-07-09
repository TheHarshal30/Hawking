import pandas as pd
import numpy as np
from collections import Counter
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.pretraining import TabNetPretrainer

from matplotlib import pyplot as plt
# %matplotlib inline

data_df = pd.read_csv('neo.csv')

useless = ['id','name','orbiting_body']
data_df = data_df.drop(useless,axis=1)

cat_cols = ['sentry_object','hazardous']
data_df[cat_cols] = data_df[cat_cols].astype(int)

lencoder = LabelEncoder()
y = pd.DataFrame(lencoder.fit_transform(data_df['hazardous']), columns=['hazardous'])
y = y.to_numpy()


X = pd.DataFrame(data_df.drop("hazardous", axis = 1))


n_samples , n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# TabNetPretrainer
unsupervised_model = TabNetPretrainer(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax',
)

import os
max_epochs = 20 if not os.getenv("CI", False) else 2

unsupervised_model.fit(
    X_train=X_train,
    eval_set=[X_test],
    max_epochs=max_epochs , patience=5,
    batch_size=1000, virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    pretraining_ratio=0.8,
)

# Make reconstruction from a dataset
reconstructed_X = unsupervised_model.predict(X_test)

unsupervised_explain_matrix, unsupervised_masks = unsupervised_model.explain(X_test)

unsupervised_model.save_model('./test_pretrain')
loaded_pretrain = TabNetPretrainer()
loaded_pretrain.load_model('./test_pretrain.zip')

clf = TabNetClassifier(optimizer_fn=torch.optim.Adam,
                       optimizer_params=dict(lr=2e-2),
                       scheduler_params={"step_size":10, # how to use learning rate scheduler
                                         "gamma":0.9},
                       scheduler_fn=torch.optim.lr_scheduler.StepLR,
                       mask_type='sparsemax' # This will be overwritten if using pretrain model
                      )

clf = TabNetClassifier()

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

clf.fit(
    X_train = X_train,
    y_train = y_train,
    eval_set=[(X_train,y_train),(X_test,y_test)],
    eval_name=['train', 'test'],
    eval_metric=['auc'],
    max_epochs=max_epochs , patience=20,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=False,
    from_unsupervised=loaded_pretrain
)

preds_valid = clf.predict_proba(X_test)
valid_auc = roc_auc_score(y_score=preds_valid[:,1], y_true=y_test)

print(valid_auc*100)