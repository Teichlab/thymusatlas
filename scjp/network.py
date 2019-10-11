from collections import Counter
from collections import defaultdict
import scanpy.api as sc
import pandas as pd
import pickle as pkl
from .cc_genes import cc_genes
from .colors import vega_20, vega_20_scanpy, zeileis_26, godsnot_64
import numpy as np
from bbknn import bbknn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns


### network derivation

# select = get_grid(bdata,n_neighbor=n_neighbor,select_per_grid = 20,scale=1)
# idata = impute_neighbor(bdata,n_neighbor=n_neighbor)
# subset = ~bdata.obs['anno_v24'].isin(["NMP"])
# select_subset = [x for x in select if x in np.where(subset)[0]]
# tfdata = new_exp_matrix(bdata,idata,select_subset,tflist=tf_lists,max_cutoff=0.1,ratio_expressed=0.005,min_disp=0.5,example_gene='CREB3L3')
# generate_gene_network(tfdata,n_neighbors=n_neighbor)
# anno_key = 'anno_v24'
# anno_uniq, anno_ratio = impute_anno(bdata,select_subset,anno_key,n_neighbor=n_neighbor)
# draw_graph(tfdata, anno_key, anno_uniq, anno_ratio,adjust=True)

def get_grid(bdata, scale=1, border=2
           ,select_per_grid=5, min_count = 2, n_neighbor = 10):
    
    import math
    from collections import defaultdict
    def shift(crd, pos):
        if (min(crd[:,pos])<0): crd[:,pos]+=abs(min(crd[:,pos]))
        if (max(crd[:,pos])<0): crd[:,pos]+=abs(max(crd[:,pos]))
    
    # get umap coordinate
    crd = bdata.obsm['X_umap'] 
    
    shift(crd,0)
    shift(crd,1)
    shift(crd,2)
    
    picture = np.zeros((scale*math.ceil(max(crd[:,0])-min(crd[:,0]))+border*scale, 
                        scale*math.ceil(max(crd[:,1])-min(crd[:,1]))+border*scale,
                        scale*math.ceil(max(crd[:,2])-min(crd[:,2]))+border*scale))

    for pos in crd*scale+border*scale/2:
        picture[math.floor(pos[0]),math.floor(pos[1]),math.floor(pos[2])]+=1
    
    plt.imshow(np.sum(picture,axis=1),vmin=10)
    plt.grid(False)
    plt.show()
    
    plt.hist(picture[picture>5],bins=100)
    plt.show()
    
    # prepare grid
    grid = defaultdict(list)
    for idx, pos in enumerate(crd*scale+border*scale/2):
        posid = '%i:%i:%i'%(math.floor(pos[0]),math.floor(pos[1]),math.floor(pos[2]))
        grid[posid].append(idx)
    
    # select grid which has more than [min_count] cells and np.min(grid_size,select_per_grid) number of representative cells from grid
    np.random.seed(0)

    select = []
    for posid in grid:
        grid_size = len(grid[posid])
        if grid_size < min_count:
            continue
        else:
            select.extend(np.random.choice(grid[posid],size=min([grid_size,select_per_grid]),replace=False))
    
    return select

def impute_neighbor(bdata,n_neighbor=10):
    from scipy.spatial import cKDTree
    from sklearn.neighbors import KDTree
    import multiprocessing as mp

    n_jobs = mp.cpu_count()
    
    # Get neighborhood structure based on 
    ckd = cKDTree(bdata.obsm["X_umap"])
    ckdout = ckd.query(x=bdata.obsm["X_umap"], k=n_neighbor, n_jobs=n_jobs)
    indices = ckdout[1]
    
    sum_list = []
    import scipy
    for i in range(0,bdata.raw.X.shape[0],10000):
        start = i
        end = min(i+10000,bdata.raw.X.shape[0])
        X_list = [bdata.raw.X[indices[start:end,i]] for i in range(n_neighbor)]
        X_sum = scipy.sparse.csr_matrix(np.sum(X_list)/n_neighbor)
        sum_list.append(X_sum)
        print(i)
        
    imputed = scipy.sparse.vstack(sum_list)
    idata = sc.AnnData(imputed)
    idata.obs = bdata.obs.copy()
    idata.var = bdata.raw.var.copy()
    idata.obsm = bdata.obsm.copy()
    idata.uns = bdata.uns.copy()
    
    return idata

def new_exp_matrix(bdata,idata,select,n_min_exp_cell = 10, min_mean=0,min_disp=.1, ratio_expressed = 0.1,example_gene='CDK1',show_filter = None, max_cutoff=0.2, tflist = None):
    
    # get genes expressed more than min_exp_cell
    detected = np.sum(bdata.raw.X>0,axis=0).A1
    
    Xnew = idata.X[select].todense()
    
    import scipy
    gdata = sc.AnnData(scipy.sparse.csr_matrix(Xnew))
    gdata.var_names = bdata.raw.var_names
    gdata.raw = sc.AnnData(Xnew)
    
    # select highly variable genes
    print('selecting hvgs...')
    result = sc.pp.filter_genes_dispersion(gdata.X,log=False,min_mean=min_mean,min_disp=min_disp)
    
    if example_gene:
        pos = np.where(gdata.var_names==example_gene)[0][0]
        plt.hist(gdata.X[:,pos].todense().A1)
        plt.show()
        print('max:',np.max(gdata.X[:,pos]))
        print('mean:',np.mean(gdata.X[:,pos]))
        print('min:',np.min(gdata.X[:,pos]))
        print('dispersions_norm:',result['dispersions_norm'][pos])
    
    if show_filter:
        x = result.means
        y = result.dispersions_norm
        c = result.gene_subset
        print(np.sum(c), 'highly variable genes are selected')
        plt.scatter(x,y)
        plt.scatter(x[c],y[c])
        plt.show()
    
    gene_info = gdata.copy()
    filter_result = result.copy()
    
    # do filter
    c1 = (result.gene_subset) # highly variable above min_disp
    c2 = (np.max(gdata.X,axis=0).todense().A1 > max_cutoff) # max expression should be above max_cutoff
    c3 = np.sum(gdata.X>max_cutoff,axis=0).A1 > ratio_expressed*len(gdata.obs_names)
    c4 = detected > n_min_exp_cell
    deg = (c1 & c3 & c4)
    gdata = gdata[:,deg].copy()
    
    # invert gene to cell
    import scipy
    cdata = sc.AnnData(scipy.sparse.csr_matrix(gdata.X.T))
    cdata.obs_names = gdata.var_names
    
    if tflist:
        tf_idx = cdata.obs_names.isin(tflist)
        tfdata = cdata[tf_idx].copy()
        return tfdata, gene_info, filter_result
    else:
        return cdata


def generate_gene_network(tfdata):
    #sc.pp.pca(tfdata)
    #sc.pp.neighbors(tfdata,metric='cosine',n_neighbors=n_neighbors)
    sc.tl.umap(tfdata,min_dist=0.7)
    sc.tl.draw_graph(tfdata,layout='fa')
    sc.tl.draw_graph(tfdata,layout='fr')
    sc.tl.draw_graph(tfdata,layout='kk')
    
def impute_anno(bdata,select, anno_key,n_neighbor=10):
    from scipy.spatial import cKDTree
    from sklearn.neighbors import KDTree
    import multiprocessing as mp

    n_jobs = mp.cpu_count()

    # Get neighborhood structure based on 
    ckd = cKDTree(bdata.obsm["X_umap"])
    ckdout = ckd.query(x=bdata.obsm["X_umap"], k=n_neighbor, n_jobs=n_jobs)
    indices = ckdout[1]
    anno_uniq = sorted(set(bdata.obs[anno_key]))
    anno_arr = np.vstack([np.array(bdata.obs[anno_key]==x).astype(int) for x in anno_uniq])
    anno_sum = np.zeros(shape=anno_arr.shape)

    for i in range(n_neighbor):
        anno_sum += anno_arr[:,indices[:,i]]

    anno_ratio = anno_sum/np.sum(anno_sum,axis=0)
    select_anno = [x in set(bdata.obs[anno_key][select]) for x in anno_uniq]
    
    return np.array(anno_uniq)[select_anno], anno_ratio[select_anno][:,select]


def draw_graph(tfdata, anno_key, anno_uniq, anno_ratio,adjust=False,z_score_cut = 2, 
               factor0 = 2, text_fontsize=10,color_dict = None, axis= 'X_draw_graph_fa'):
    import seaborn as sns
    palette1 =    ["#023fa5", "#7d87b9", "#bec1d4", "#d6bcc0", "#bb7784", "#8e063b", "#4a6fe3",
        "#8595e1", "#b5bbe3", "#e6afb9", "#e07b91", "#d33f6a", "#11c638", "#8dd593",
        "#c6dec7", "#ead3c6", "#f0b98d", "#ef9708", "#0fcfc0", "#9cded6", "#d5eae7",
        "#f3e1eb", "#f6c4e1", "#f79cd4",
        '#7f7f7f', "#c7c7c7", "#1CE6FF", "#336600"]

    palette2 = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2',  # '#7f7f7f' removed grey
        '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896',
        '#c5b0d5', '#c49c94', '#f7b6d2',  # '#c7c7c7' removed grey
        '#dbdb8d', '#9edae5',
        '#ad494a', '#8c6d31']  # manual additions

    
    if len(anno_uniq)> 28:
        palette = godsnot_64
    elif len(anno_uniq)> 20:
        palette = palette1
    else:
        palette = palette2
        
    if color_dict:
        palette = [color_dict[x] for x in anno_uniq]
        
    Cr = np.corrcoef(np.vstack([tfdata.X.todense(),anno_ratio]))
    Cr_an = Cr[len(tfdata):,:len(tfdata)]
    color_anno = np.argmax(Cr_an,axis=0)
    
    C = tfdata.uns['neighbors']['connectivities'].todense()
    #C = Cr[:len(tfdata),:len(tfdata)]
    pairs = np.where(C>0.3)

    from sklearn import preprocessing

    Cr_scaled = preprocessing.scale(Cr_an)
    Cr_scaled = Cr_an
    #dot_size0 = np.choose(color_anno,Cr_scaled)
    dot_size0 = np.array([Cr_scaled[j,i] for i,j in enumerate(color_anno)])
    
    colors = np.array([palette[i] if s >z_score_cut else 'gray' for i,s in zip(color_anno,dot_size0)])
    
    from adjustText import adjust_text

    show = True
    gamma = 2
    dot_size = dot_size0**(gamma)

    x = tfdata.obsm[axis][:,0]
    y = tfdata.obsm[axis][:,1]
    n = list(tfdata.obs_names)

    plt.figure(figsize=(8,8))
    plt.scatter(x,y,c=colors,s=factor0*dot_size) #50*size

    # draw pair to pair lines
    for p1, p2 in zip(pairs[0],pairs[1]):
        plt.plot([x[p1],x[p2]],[y[p1],y[p2]],c='lightgrey',alpha=0.3,zorder=-1,
                linewidth=2*C[p1,p2]**gamma)

    # draw_label
    for i, label in enumerate(anno_uniq):
        plt.scatter(0,0,s=0,label=label,
                    c=palette[i],zorder=1,alpha=1.0,
                    linewidth=0)
    # add_names
    if show != False:   
        texts = []
        for i, txt in enumerate(n):
            if txt in anno_uniq:
                texts.append(plt.text(x[i],y[i],txt,fontsize=text_fontsize*1.2,color=color_dict[txt]))
            else:
                texts.append(plt.text(x[i],y[i],txt,fontsize=text_fontsize))

    lgnd = plt.legend(loc=(1.1,0), scatterpoints=1, fontsize=10)
    for handle in lgnd.legendHandles:
        handle.set_sizes([30.0])
        handle.set_alpha(1) 

    plt.xticks([], [])
    plt.yticks([], [])
    plt.grid(False)
    if adjust == True:
        adjust_text(texts,only_move={'text':'y'})