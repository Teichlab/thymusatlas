import numpy as np
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import scanpy.api as sc

# Logistic model methods
    
def generate_training_X(adata,ct_key,select_num=200,exclude=[]):
    ad_model = adata
    ct_list = [x for x in set(ad_model.obs[ct_key])]
    select = np.hstack([np.random.choice(np.where(ad_model.obs[ct_key]==ct)[0],select_num) for ct in ct_list if ct not in exclude])
    ad_model = ad_model[select]
                                         
    return ad_model


def logistic_model(data, cell_types, sparsity=0.2, fraction=0.2, penalty='l2'):
    X = data
    X_train, X_test, y_train, y_test = \
    train_test_split(X, cell_types, test_size=fraction)
    lr = LogisticRegression(penalty=penalty, C=sparsity)
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)
    return y_prob, y_test, lr

from sklearn import metrics
from scanpy.plotting.palettes import *

def plot_roc(y_prob, y_test, lr):
    aucs =[]
    if len(lr.classes_)<21: colors = default_20
    elif len(lr.classes_)<27: colors = default_26
    else: colors = default_64
    for i, cell_type in enumerate(lr.classes_):
        fpr, tpr, _ = metrics.roc_curve(y_test == cell_type, y_prob[:, i])
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)
        plt.plot(fpr, tpr, c=colors[i], lw=2, label = cell_type)
    plt.plot([0, 1], [0, 1], color='k', ls=':')
    plt.legend(loc=(1,0))
    min_auc = np.min(aucs)
    plt.title("Min AUC: %.3f"%(min_auc))
    return(min_auc)


def transfer_annotation_jp(input_adata,y_id,output_adata,y_out,select_num=200,log=None,exclude=[],raw=True):
    ad_model = generate_training_X(input_adata,y_id,select_num=select_num,exclude=exclude)
    #sc.pl.umap(ad_model,color=y_id)

    if raw==False:
        X_model = ad_model.X
    else:
        X_model = ad_model.raw.X
    y_model = ad_model.obs[y_id]
    if raw==False:
        X_predict = output_adata.X
    else:
        X_predict = output_adata.raw.X

    print("creating lr model...")
    y_prob, y_test, lr =  logistic_model(X_model, y_model, sparsity=0.2, fraction=0.2, penalty='l2')
    plot_roc(y_prob, y_test, lr)
    
    print("making lr prediction...")
    Lout = {}
    if log:
        Lout['log'] = log
    Lout['classes'] = lr.classes_
    Lout['predict'] = lr.predict(X_predict)
    Lout['predict_proba'] = lr.predict_proba(X_predict)
    
    print("updating lr to adata...")
    output_adata.obs[y_out] = Lout['predict']
    
    return lr

def update_label(from_adata,from_label,to_adata,old_label,new_label,
                 exclude=None,include=None,replace=False,unknown=None,keep_replaced=True):
    if old_label not in to_adata.obs.columns:
        to_adata.obs[old_label] = 'unknown'
    if exclude:
        ON = {O:N for O,N in zip(from_adata.obs_names,from_adata.obs[from_label]) if N not in exclude}
    elif include:
        ON = {O:N for O,N in zip(from_adata.obs_names,from_adata.obs[from_label]) if N in include}
    else:
        ON = {O:N for O,N in zip(from_adata.obs_names,from_adata.obs[from_label])}
        
 
    if unknown:
        new_anno = [ON[O] if ((O in ON) & (N == unknown)) else N for O,N in zip(to_adata.obs_names,to_adata.obs[old_label])]
    else:
        new_anno = [ON[O] if (O in ON) else N for O,N in zip(to_adata.obs_names,to_adata.obs[old_label])]
    
    if new_label in to_adata.obs.columns:
        if replace==False:
            if keep_replaced==True:
                raise SystemError
            else:
                to_adata.obs[new_label] = new_anno
        elif replace==True:
            i =  1
            while True:
                new_key = new_label+'.replaced.'+str(i)
                if new_key in to_adata.obs.columns:
                    i+=1
                    continue
                else:
                    to_adata.obs[new_key] = list(to_adata.obs[new_label])
                    del to_adata.obs[new_label]
                    to_adata.obs[new_label] = new_anno
                    break
    else:
        to_adata.obs[new_label] = new_anno


def get_common_var_raw(a,b):
    common = sorted(list(set(a.raw.var_names).intersection(set(b.raw.var_names))))
    list_a_names = list(a.raw.var_names)
    list_b_names = list(b.raw.var_names)
    a_index = np.array([list_a_names.index(x) for x in common])
    b_index = np.array([list_b_names.index(x) for x in common])
    print('calculating a...')
    a_new_X = a.raw.X[:,a_index]
    print('calculating b...')
    b_new_X = b.raw.X[:,b_index]
    a_new = sc.AnnData(a_new_X,obs = a.obs)
    a_new.obsm = a.obsm
    a_new.var_names = common
    b_new = sc.AnnData(b_new_X,obs = b.obs)
    b_new.obsm = b.obsm
    b_new.var_names = common
    return a_new,b_new

def fill_columns(a_new,b_new):
    a_col_list = list(a_new.obs.columns)
    b_col_list = list(b_new.obs.columns)
    for obs_item in a_col_list:
        if obs_item not in b_col_list:
            b_new.obs[obs_item] = 'nan'
    for obs_item in b_col_list:
        if obs_item not in a_col_list:
            a_new.obs[obs_item] = 'nan'