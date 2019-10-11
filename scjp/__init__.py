from collections import Counter
from collections import defaultdict
import scanpy.api as sc
import scrublet as scr
import pandas as pd
import pickle as pkl
from .cc_genes import cc_genes
from .colors import vega_20, vega_20_scanpy, zeileis_26, godsnot_64
from .markers import find_markers, show_marker_plot
from .model import generate_training_X
import numpy as np
from bbknn import bbknn

import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import re
import glob
import os
import sys
from geosketch import gs

Folder = '/mnt/thymus_atlas/scjp'

tf_genes = pd.read_csv(Folder+"/D05_Human_TF.txt",sep='\t')
tf_lists = list(tf_genes.Symbol)

def merge_matrix(ad,obskeys = None,use_raw = False,keep_only_mutual=False):
    '''merge matrix stored in ad
    ad: dictionary of anndata to merge
    obskeys: list to merge within anndata
    use_raw: if True, merge from .raw.X'''
    
    smp_list = list(ad.keys())
    obs_dict = defaultdict(list)
    obs_names = []
    
    for smp in smp_list:
        ad[smp].obs['name'] = smp
    
    if not obskeys:
        obskey_list = []
        obskeys = []
        for sample in smp_list:
            obskey_list.extend(list(ad[sample].obs.columns))
        for (obskey, number) in Counter(obskey_list).items():
            if number == len(smp_list):
                obskeys.append(obskey)
            else:
                if keep_only_mutual:
                    pass
                else:
                    for sample in smp_list:
                        if obskey not in ad[sample].obs.columns:
                            ad[sample].obs[obskey]='n/a'
                    obskeys.append(obskey)
                               
    for sample in smp_list:
        obs_names.extend(list(ad[sample].obs_names))
        for key in obskeys:   
            obs_dict[key].extend(list(ad[sample].obs[key]))
    
    from scipy.sparse import vstack
    if use_raw == True:
        stack = vstack([ad[x].raw.X for x in smp_list]) # stack data
        adata = sc.AnnData(stack, var = ad[smp_list[0]].raw.var)
    else:
        stack = vstack([ad[x].X for x in smp_list]) # stack data
        adata = sc.AnnData(stack, var = ad[smp_list[0]].var)
      
    adata.obs_names = obs_names
    print(len(adata))
    for obs_col in obs_dict:
        print(obs_col)
        adata.obs[obs_col] = obs_dict[obs_col]
    return adata

def write(adata,version,name):
    '''write adata into [name]'''
    name = version + name
    sc.write(name,adata)
    print("_".join(name.split(".")) + " = '%s'"%name)
    
def jp_save_fig(version,figcount,fig_format='pdf',fig_folder='11_Figs'):
    
    plt.savefig('%s/%s%s.%s'%(fig_folder,version,figcount,fig_format),bbox_inches='tight',format=fig_format,dpi=300)
    print('%s/%s%s.pdf'%(fig_folder,version,figcount))
    
def doublet(adata, key='Sample'):
    '''detecting doublet using scrublet per key'''
    doublet = []
    for filename in set(adata.obs[key]):
        print(filename)
        sdata = adata[adata.obs[key] == filename].copy()
        scrub = scr.Scrublet(sdata.X)
        doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False)
        doublet.extend([(x,y,z) for x,y,z in zip(sdata.obs_names,doublet_scores,predicted_doublets)])
    doublet_score = {x:y for (x,y,z) in doublet}
    doublet_predict = {x:z for (x,y,z) in doublet}
    adata.obs['doublet_score'] = [doublet_score[obs_name] for obs_name in list(adata.obs_names)]
    adata.obs['doublet_predict'] = [doublet_predict[obs_name] for obs_name in list(adata.obs_names)]

def get_sketch(adata,key,folds=10,how='pd',min_num_per_key=500,start='filter'):
    '''geometric sketching based on diffusion map and pca
    folds: folds to subsample
    min_num_per_key: minimun number to sample'''
    sketch_index = []
    for smp in set(adata.obs[key]):
        print(smp)
        c = adata.obs[key] == smp
        
        if start=='filter':
            sdata = get_subset(adata,c)
        else:        
            sdata = adata[c]
            sc.pp.filter_genes_dispersion(sdata)
            sc.pp.pca(sdata)
        sc.pp.neighbors(sdata)
        sc.tl.diffmap(sdata)

        N = np.max([np.int(np.sum(c)/folds),np.min([min_num_per_key,np.sum(c)])])
        print(N)
        if how =='pd':
            set1 = set(sdata.obs_names[gs(sdata.obsm['X_diffmap'],N,replace=True)])
            set2 = set(sdata.obs_names[gs(sdata.obsm['X_pca'][:,:50],N,replace=True)])
            sketch_index.extend(list(set1.union(set2)))
        elif how =='p':
            set2 = set(sdata.obs_names[gs(sdata.obsm['X_pca'][:,:50],N,replace=True)])
            sketch_index.extend(list(set2))
        elif how =='d':
            set1 = set(sdata.obs_names[gs(sdata.obsm['X_diffmap'][:,:20],N,replace=True)])
            sketch_index.extend(list(set1))
        else:
            raise SystemError
    return(sketch_index)

    
def regress_batch_v2(adata,batch_key,confounder_key):
    '''batch regression tool
    batch_key=list of observation categories to be regressed out
    confounder_key=list of observation categories to be kept
    returns ndata with corrected X'''

    from sklearn.linear_model import Ridge
    
    dummy = pd.get_dummies(adata.obs[batch_key+confounder_key],drop_first=False)
    X_exp = adata.X # scaled data
    if scipy.sparse.issparse(X_exp):
        X_exp = X_exp.todense()
    LR = Ridge(fit_intercept=False,alpha=1.0)
    LR.fit(dummy,X_exp)

    if len(batch_key)>1:
        batch_index = np.logical_or.reduce(np.vstack([dummy.columns.str.startswith(x) for x in batch_key]))
    else:
        batch_index = np.vstack([dummy.columns.str.startswith(x) for x in batch_key])[0]
    
    dm = np.array(dummy)[:,batch_index]
    X_explained = dm.dot(LR.coef_[:,batch_index].T)
    X_remain = X_exp - X_explained
    ndata = sc.AnnData(X_remain)
    ndata.obs = adata.obs
    ndata.var = adata.var
    return ndata, X_explained

def regress_iter(adata,batch_key,confounder_key,bbknn_key,scale=True, approx = True,n_pcs=50):
    if scale == True:
        sc.pp.scale(adata,max_value=10)
    ndata, X_explained = regress_batch_v2(adata,batch_key=batch_key,confounder_key=confounder_key)
    sc.pp.pca(ndata)
    bbknn(ndata, batch_key = bbknn_key,n_pcs=n_pcs, approx=approx)
    return ndata, X_explained

def run_pca_bbknn_umap(ad,level_key,bbknn_key,marker_dict,
                       resolution=0.02,start = 'leiden',select=False,
                      thres=0.95,min_drop_cut=0.5,show=True, how='almost',
                      min_cluster_num = 200):
    '''
    run pca, bbknn, umap and clustering to find good low-rescluster with markers
    how = [almost, any, all] for marker_found function
    '''
    adata = ad[level_key]
    if start == 'pca':
        sc.pp.pca(adata)
        bbknn(adata,batch_key=bbknn_key,approx=False)
        sc.tl.umap(adata)
    if start in 'leiden,pca'.split(','):
        sc.tl.leiden(adata,resolution=resolution)
        adata.obs[level_key] = list(adata.obs['leiden'])
    if start in 'leiden,pca,mks'.split(','):
        if show:
            sc.pl.umap(adata,color='leiden')
        if len(set(adata.obs['leiden']))<2:
            print('clustering not enough')
            return False,None
        elif np.min([x for x in Counter(adata.obs['leiden']).values()]) < min_cluster_num:
            print('clustering resolution too high')
            return False,None
        else:
            if select:
                ndata = generate_training_X(adata,'leiden',select_num=select)
            else:
                ndata = adata
            mks = find_markers(ndata,'leiden',thres=thres,min_drop_cut=min_drop_cut)
            if show:
                show_marker_plot(ndata,'leiden',mks,toshow=5)
            go = marker_found(mks,how=how)
            if go:
                commit_level(adata,level_key,mks,marker_dict)
            else:
                print('marker not found')
            return go, mks
    else:
        'start accepts either pca or leiden'
        raise SystemError

def marker_found(mks,how='any'):
    if how=='any':
        c1 = len([keys for keys,values in mks.items() if len(values)>0]) > 0 
    elif how == 'some':
        c0 = len([keys for keys,values in mks.items() if len(values)>0]) >= 3
        c2 = len([keys for keys,values in mks.items() if len(values)>0]) >= (len(mks.keys())-1)
        c1 = c0|c2
    elif how=='all':
        c1 = len([keys for keys,values in mks.items() if len(values)>0]) == len(mks.keys())
    elif how=='almost':
        c1 = len([keys for keys,values in mks.items() if len(values)>0]) >= (len(mks.keys())-1)
    else:
        print('Error: print how not in any, all, alomst')
        raise SystemExit
    return c1

def commit_level(adata,level_key,mks,marker_dict):
    for leiden_clst in set(adata.obs[level_key]):
        to_merge = np.array(adata.obs[level_key].copy(),dtype=object)
        if len(mks[leiden_clst]) >0:
            final_key = level_key+"_"+leiden_clst
            marker_dict[final_key] = mks[leiden_clst]
        else:
            to_merge[adata.obs[level_key]==leiden_clst]='M'
        adata.obs[level_key] = to_merge
                
def expand_level_copy(ad,level_key):
    adata = ad[level_key]
    for leiden_clst in set(adata.obs[level_key]):
        final_key = level_key+"_"+leiden_clst
        print(final_key)
        ad[final_key] = adata[adata.obs[level_key]==leiden_clst].copy()
        
def summary(ad):
    anno = np.zeros(len(ad['0']),dtype=object)
    final_clusters = []
    for k in ad.keys():
        if np.sum([x.startswith(k) for x in ad.keys()]) ==1:
            final_clusters.append(k)
    for k in final_clusters:
        anno[ad['0'].obs_names.isin(ad[k].obs_names)] =k
    return anno
        
def walk_cluster(ad,marker_dict,tried,bbknn_key,
                 leiden_walk=[0.02,0.05], thres=0.95, min_drop_cut=0.5,
                 select=False, show=False, how='almost', 
                 final_limit_num=8, min_num_split=500):
    go = False
    processed = set(['_'.join(x.split('_')[:-1]) for x in marker_dict.keys()])
    to_process = [level_key for level_key in list(ad.keys()) if level_key not in processed.union(set(tried))]

    print(to_process)
    for level_key in to_process:
        print(level_key)
        if len(level_key.split("_")) > final_limit_num:
            print('level too deep')
            continue
        if len(ad[level_key])<min_num_split:
            print('subset too small')
            continue
        for resolution in leiden_walk:
            print(resolution)
            result = run_pca_bbknn_umap(ad,level_key,bbknn_key,marker_dict,
                                             start='leiden',resolution=resolution,
                                            thres=thres,min_drop_cut=min_drop_cut,
                                            select=select,show=show,how=how)
            if result[0]:
                print('marker found at '+str(resolution))
                go = True
                expand_level_copy(ad,level_key)
                break
        tried.append(level_key)
    return(go)
 
# General procedure
    
def sc_process(adata,pid = 'fpnul',n_pcs=50):
    if 'f' in pid:
        sc.pp.filter_genes_dispersion(adata)
    if 's' in pid:
        sc.pp.scale(adata,max_value=10)
    if 'p' in pid:
        sc.pp.pca(adata)
    if 'n' in pid:
        sc.pp.neighbors(adata,n_pcs=n_pcs)
    if 'u' in pid:
        sc.tl.umap(adata)
    if 'l' in pid:
        sc.tl.leiden(adata)
    
def bbknn_umap(adata,batch_key,n_pcs,cluster=False,n_neighbors=3):
    bbknn(adata,batch_key=batch_key,n_pcs=n_pcs,approx=False,neighbors_within_batch=n_neighbors)
    if cluster:
        sc.tl.leiden(adata)
    sc.tl.umap(adata)
    
def umap(adata,name=None):
    sc.tl.umap(adata)
    if name:
        adata.obsm['X_umap_'+name] = adata.obsm['X_umap'].copy()
        
def umap_show(adata,feature,feature_name= None):
    if feature_name:
        adata.obs[feature_name] = feature
        sc.pl.umap(adata,color=feature_name,color_map='OrRd')
    else:
        adata.obs['show'] = feature
        sc.pl.umap(adata,color='show',color_map='OrRd')

def remove_geneset(adata,geneset):
    adata = adata[:,~adata.var_names.isin(list(geneset))].copy()
    return adata

def is_cycling(adata,cc_genes=cc_genes,cut_off=0.4):
    X = np.mean(adata.raw.X[:,adata.raw.var_names.isin(cc_genes)],axis=1)
    plt.hist(X)
    adata.obs['Cycle_score'] = X
    adata.obs['isCycle'] = X>cut_off
    
def get_subset(idata, select, cc_genes=cc_genes, log=False,raw=True):
    if raw:
        adata = sc.AnnData(idata[select].raw.X)
        adata.var = idata.raw.var
    else:
        adata = sc.AnnData(idata[select].X)
        adata.var = idata.var
    adata.obs = idata.obs[select]
    adata.raw = adata.copy()
    #adata.X = scipy.sparse.csr_matrix(np.exp(adata.X.todense())-1)
    sc.pp.filter_genes_dispersion(adata,log=log)
    if log:
        sc.pp.log1p(adata)
    sc.pp.scale(adata,max_value=10)
    if len(cc_genes)>0:
        remove_geneset(adata,cc_genes)
    sc.pp.pca(adata,n_comps = np.min([50,adata.X.shape[0],adata.X.shape[1]]))
    return adata

def get_raw(idata, cc_genes=cc_genes, log=False):
    adata = sc.AnnData(idata.raw.X)
    adata.var = idata.raw.var
    adata.obs = idata.obs
    adata.raw = adata.copy()
    #adata.X = scipy.sparse.csr_matrix(np.exp(adata.X.todense())-1)
    sc.pp.filter_genes_dispersion(adata,log=log)
    if log:
        sc.pp.log1p(adata)
    sc.pp.scale(adata,max_value=10)
    if len(cc_genes)>0:
        remove_geneset(adata,cc_genes)
    sc.pp.pca(adata,n_comps = np.min([50,adata.X.shape[0],adata.X.shape[1]]))
    return adata

def output_matrix_Seurat(adata,version,name,use_raw=False):
    
    from scipy.io import mmwrite
    
    if use_raw:
        X = adata.raw.X
        mmwrite(version+name+'.mtx',X)
        adata.obs.to_csv(version+name+'.meta.csv')
        adata.raw.var.to_csv(version+name+'.var.csv')
    else:
        X = adata.X
        mmwrite(version+name+'.mtx',X)
        adata.obs.to_csv(version+name+'.meta.csv')
        adata.var.to_csv(version+name+'.var.csv')

def us(adata,gene,groups=None, show=False, **kwargs):
    if ',' in gene:
        gene = gene.split(',')
    if groups:
        sc.pl.umap(adata,color=gene,color_map='OrRd',groups=groups, show=show, **kwargs)
    else:
        sc.pl.umap(adata,color=gene,color_map='OrRd',show=show, **kwargs)
        
class annotater():
    '''
    create de novo annotation onto adata
    '''
    
    def __init__(self,adata,new_label_name,old_label=None):
    
        if old_label:
            adata.obs[new_label_name] = adata.obs[old_label]
        else:
            adata.obs[new_label_name] = 'unknown'
        arr = np.array(adata.obs[new_label_name],dtype=object)
        self.new_label = arr
        self.new_label_name = new_label_name
        
    def update(self,adata,obskey,select,label_name):
        if ',' in select:
            label_condition = adata.obs[obskey].isin(select.split(','))
        else:
            label_condition = adata.obs[obskey]==select
        self.new_label[label_condition] = label_name
        adata.obs[self.new_label_name] = self.new_label
    
    def update_condi(self,adata,label_condition,label_name):

        self.new_label[label_condition] = label_name
        adata.obs[self.new_label_name] = self.new_label
        