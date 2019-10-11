import numpy as np
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt


class marker():
    
    def __init__(self,adata,label,**kwargs):
        self.adata = adata
        self.label = label
        self.mks = find_markers(adata,label,**kwargs)
    
    def plot_marker(self,**kwargs):
        return show_marker_plot(self.adata,self.label,self.mks,**kwargs)
    
    def show_marker(self,celltype,**kwargs):
        return show_marker(self.mks,celltype,**kwargs)

# Marker gene analysis
def calculate_markers(adata,group_key,exclude = None, cnt_cut=0, tot_cnt_cut=0):

    # group_key = 'louvain'
    # tot_cnt_cut = minimum cell number for ct, default 0, no cut here
    # cnt_cut = minumum cell number of expressing that gene per cell, otherwise, 0

    ctlist = np.array(sorted(list(set(adata.obs[group_key]))))
    
    if exclude:
        ctlist = [x for x in ctlist if x not in exclude]
    
    mtx = adata.raw
    len_gene = mtx.X.shape[1]

    mean_dt = {}
    cnt_dt = {}
    drop_dt = {}

    for ct in ctlist:
        ct_cond = np.array(adata.obs[group_key]==ct)
        if np.sum(ct_cond)<tot_cnt_cut:
            continue
        #print(ct)
        len_ct = np.sum(ct_cond)
        mtx_ct = mtx.X[ct_cond]
        pos_count = Counter(mtx_ct.nonzero()[1])
        cnt_arr = np.zeros(len_gene)
        for pos in pos_count:
            cnt_arr[pos] = pos_count[pos]
        cnt_dt[ct] = cnt_arr
        cnt_mask = (cnt_arr<cnt_cut)
        drop_dt[ct] = cnt_arr/len_ct
        mean_arr = np.mean(mtx_ct,axis=0).A1
        mean_arr[cnt_mask] = 0
        mean_dt[ct] = mean_arr
    
    cdm_out = {"cnt":cnt_dt, "drop":drop_dt, "mean":mean_dt}
    return cdm_out


def find_markers_groups(adata, cdm_out, groups, thres = 0.2, min_exp_cut=0.1, min_cnt_cut = 5):
    # Select marker gene (multiple)
     
    cnt_dt = cdm_out["cnt"]
    drop_dt = cdm_out["drop"]
    mean_dt = cdm_out["mean"]
    
    ctlist = np.array(sorted(list(drop_dt.keys())))
    marker_group = []
    thres = thres

    for k,gname in enumerate(adata.raw.var_names):  
        x1 = []
        y1 = []
        ct_selected = []
        for ct in ctlist:
            if cnt_dt[ct][k] < min_cnt_cut:
                continue
            x1.append(drop_dt[ct][k]) # get expratio
            y1.append(mean_dt[ct][k]) # get expmean
            ct_selected.append(ct)
        x1 = np.array(x1)
        y1 = np.array(y1)
        if len(y1) == 0:
            continue
        y1 = y1/np.max(y1)
        # sort expratio, expmean, celltypelist
        
        is_group = np.array([x in (groups) for x in ct_selected])
        try: max_y1 = np.max(y1[~is_group])
        except: max_y1 = 0
        try: min_y1 = np.min(y1[is_group])
        except: min_y1 = 0
        
        dff = min_y1-max_y1
        cond = dff>thres
            
        if cond:
            chgpos = (np.where(cond)[0][0])+1
            out = 1
            for i,ct_high in enumerate(list(groups)):
                if mean_dt[ct_high][k] < min_exp_cut:
                    out = 0
            if out == 1:
                marker_group.append((gname,dff)) 
    return  marker_group


def find_markers_multiple(adata, cdm_out, thres = 0.2, min_mean_cut=0.2, min_drop_cut=0.2, min_cnt_cut = 0):
    # Select marker gene (multiple)
    
    cnt_dt = cdm_out["cnt"]
    drop_dt = cdm_out["drop"]
    mean_dt = cdm_out["mean"]
    
    ctlist = np.array(sorted(list(drop_dt.keys())))
    marker_multiple = defaultdict(list)
    thres = thres

    for k,gname in enumerate(adata.raw.var_names):  
        x1 = []
        y1 = []
        ct_selected = []
        for ct in ctlist:
            if cnt_dt[ct][k] < min_cnt_cut:
                continue
            x1.append(drop_dt[ct][k]) # get expratio
            y1.append(mean_dt[ct][k]) # get expmean
            ct_selected.append(ct)
            
        x1 = np.array(x1)
        y1 = np.array(y1)
        if len(y1) == 0:
            continue
        y1 = y1/np.max(y1)
        # sort expratio, expmean, celltypelist
        idx = np.argsort(y1)
        x1 = x1[idx]
        y1 = y1[idx]
        ctlist_sorted = np.array(ct_selected)[idx]
        
        dff = (np.diff(y1))
        cond = dff>thres

        if any(cond):
            chgpos = (np.where(cond)[0][0])
            ct_high = ctlist_sorted[-1]
            if mean_dt[ct_high][k] < min_mean_cut:
                pass
            elif drop_dt[ct_high][k] < min_drop_cut:
                pass
            else:
                marker_multiple[ct_high].append((gname,dff[chgpos]))   
    return  marker_multiple

def find_markers_single(adata, cdm_out, thres = 0.2, min_mean_cut=0.2, min_drop_cut=0.2, min_cnt_cut = 0):
    # Select marker gene (single)
  
    cnt_dt = cdm_out["cnt"]
    drop_dt = cdm_out["drop"]
    mean_dt = cdm_out["mean"]
    
    ctlist = np.array(sorted(list(drop_dt.keys())))
    marker_single = defaultdict(list)

    for k,gname in enumerate(adata.raw.var_names):  
        x1 = []
        y1 = []
        ct_selected = []
        cond = False
        for ct in ctlist:
            if cnt_dt[ct][k] < min_cnt_cut:
                continue
            x1.append(drop_dt[ct][k]) # get expratio
            y1.append(mean_dt[ct][k]) # get expmean
            ct_selected.append(ct)
        if len(y1)==0:
            continue
        else:
            y1 = y1/np.max(y1)
            if len(y1)==1:
                if y1[0]>thres:
                    if mean_dt[ct_selected[0]][k] < min_mean_cut:
                        pass
                    elif drop_dt[ct_selected[0]][k] < min_drop_cut:
                        pass
                    else:
                        marker_single[ct_selected[0]].append((gname,y1[0]))
                else:
                    pass
            else:
                x1 = np.array(x1)
                y1 = np.array(y1)
                # sort expratio, expmean, celltypelist
                idx = np.argsort(y1)
                x1 = x1[idx]
                y1 = y1[idx]    
                ctlist_sorted = np.array(ct_selected)[idx]

                dff = (np.diff(y1))
                cond = dff[-1]>thres

                if cond==True:
                    ct_high = ctlist_sorted[-1] 
                    if mean_dt[ct_high][k] < min_mean_cut:
                        pass
                    elif drop_dt[ct_high][k] < min_drop_cut:
                        pass
                    else:
                        marker_single[ct_high].append((gname,dff[-1]))
    return marker_single 

def find_markers_negative(adata, cdm_out, ctlist = None, thres = 0.2, min_exp_cut=0.1):
    # Select marker gene (single)
  
    cnt_dt = cdm_out["cnt"]
    drop_dt = cdm_out["drop"]
    mean_dt = cdm_out["mean"]
    
    if ctlist == None:
        ctlist = np.array(sorted(list(drop_dt.keys())))
        
    marker_single = defaultdict(list)
    thres = thres

    for k,gname in enumerate(adata.raw.var_names):  
        x1 = []
        y1 = []
        ct_selected = []
        cond = False
        max_value = 0.0
        for ct in ctlist:
            max_value = np.max([max_value, mean_dt[ct][k]])
            x1.append(drop_dt[ct][k]) # get expratio
            y1.append(mean_dt[ct][k]) # get expmean
            ct_selected.append(ct)
        if max_value < min_exp_cut:
            continue
        else:
            y1 = y1/np.max(y1)
            x1 = np.array(x1)
            y1 = np.array(y1)
            # sort expratio, expmean, celltypelist
            idx = np.argsort(y1)
            x1 = x1[idx]
            y1 = y1[idx]
            ctlist_sorted = np.array(ct_selected)[idx]

            dff = (np.diff(y1))
            if gname == 'ENSG00000168685':
                print(ctlist_sorted)
                print(y1)
                print(dff)
            cond = dff[0]>thres

            if cond==True:
                ct_high = ctlist_sorted[0] 
                marker_single[ct_high].append((gname,dff[0]))
    for ct in list(marker_single.keys()):
        marker_single[ct] = sorted([(geneID,drop) for geneID,drop in marker_single[ct]],key=lambda x:-x[1])
    return marker_single 

def find_markers(adata,groupby,scanpy=None,single=True,thres=0.2,min_mean_cut=0.2, min_drop_cut=0.2,min_cnt_cut=10):
    adata.uns['cdm_'+groupby]=calculate_markers(adata,groupby) 
    if single==False:
        mks=find_markers_multiple(adata,adata.uns['cdm_'+groupby],
                                  thres=thres,min_mean_cut=min_mean_cut, 
                                  min_drop_cut=min_drop_cut, min_cnt_cut=min_cnt_cut)
    else:
        mks=find_markers_single(adata,adata.uns['cdm_'+groupby],
                                thres=thres,min_mean_cut=min_mean_cut, 
                                min_drop_cut=min_drop_cut, min_cnt_cut=min_cnt_cut)
    if scanpy:
        sc.tl.rank_genes_groups(adata,groupby)
        adata.uns['rnk']=pd.DataFrame(adata.uns['rank_genes_groups']['names'])
    return mks

def show_marker(marker_output, celltype=None, toshow=40, result=False):
    if celltype == None:
        if result == True:
            return sorted([(geneID,drop) for geneID,drop in marker_output],key=lambda x:-x[1])[0:toshow]
        else:
            print(sorted([(geneID,drop) for geneID,drop in marker_output],key=lambda x:-x[1])[0:toshow])
    else:
        if result == True:
            return sorted([(geneID,drop) for geneID,drop in marker_output[celltype]],key=lambda x:-x[1])[0:toshow]
        else:
            print(sorted([(geneID,drop) for geneID,drop in marker_output[celltype]],key=lambda x:-x[1])[0:toshow])

def show_marker_plot(adata,anno_key,mks,toshow=10,T=False):
    mklist = []
    ct_list = sorted(adata.uns['cdm_'+anno_key]['cnt'].keys())
    #ct_list = sorted(ct_list,key=lambda x:int(x))
    report_mks = {}
    
    for ct in ct_list:
        
        try:
            ct_mks = [x[0] for x in show_marker(mks,celltype=ct,result=True)][0:toshow]
        except:
            ct_mks = [' ']
        mklist.extend(ct_mks)
        mklist.append(' ')
        report_mks[ct] = ct_mks
        
    if T:
        draw_marker_blob_T(adata,adata.uns['cdm_'+anno_key],mklist,ctlist=ct_list)
    else:
        draw_marker_blob(adata,adata.uns['cdm_'+anno_key],mklist,ctlist=ct_list,min_drop_ratio=0.0)
    return(report_mks)



def draw_marker(adata, gene, cdm_out, min_cnt_cut = 0):
    

    cnt_dt = cdm_out["cnt"]
    drop_dt = cdm_out["drop"]
    mean_dt = cdm_out["mean"]
    
    ctlist = np.array(sorted(list(drop_dt.keys())))
    k = np.where(adata.raw.var_names == gene)[0][0]
    x1 = []
    y1 = []
    ct_selected = []
    for ct in ctlist:
        if cnt_dt[ct][k] < min_cnt_cut:
            continue
        x1.append(drop_dt[ct][k])
        y1.append(mean_dt[ct][k])
        ct_selected.append(ct)
    plt.figure(figsize=(5,5))
    plt.scatter(x1,y1)
    for i, ct in enumerate(ct_selected):
        plt.annotate(ct,(x1[i],y1[i]),fontsize=10)
    plt.title(gene)
    plt.xlim(0,1)
    plt.xlabel("ratio cells detected")
    plt.ylabel("mean exp")
    plt.grid(False)
    plt.show()
    
    return(x1,y1,ct_selected)

def draw_marker_v2(adata, gene, anno_key, cs, show=True):
    
    from adjustText import adjust_text

    cdm_out = adata.uns['cdm_%s'%(anno_key)]
    
    cnt_dt = cdm_out["cnt"]
    drop_dt = cdm_out["drop"]
    mean_dt = cdm_out["mean"]
    
    ctlist = np.array([x for x in sorted(list(drop_dt.keys())) if x in set(adata.obs[anno_key])])
    
    k = np.where(adata.raw.var_names == gene)[0][0]
    x1 = []
    y1 = []
    ct_selected = []
    cls = []
    for ct in ctlist:
        if cnt_dt[ct][k] < 0:
            continue
        x1.append(drop_dt[ct][k])
        y1.append(mean_dt[ct][k])
        ct_selected.append(ct)
        cls.append(cs[ct])
    plt.figure(figsize=(5,5))
    plt.scatter(x1,y1,color=cls)

    texts = []
    for i, txt in enumerate(ct_selected):
        texts.append(plt.text(x1[i],y1[i],txt,fontsize=10,color=cs[txt]))
        
    plt.title(gene)
    plt.xlim(0,max(x1)+0.1)
    plt.xlabel("ratio cells detected")
    plt.ylabel("mean exp")
    plt.grid(False)
    adjust_text(texts,only_move={'texts':'xy'},lim=3)
    if show:
        plt.show() 

def draw_marker_blob_v2(adata,cdm_out,genes,ctlist = None,fontsize=10,
                     figscale=0.25,save=None, show=True,
                     min_drop_ratio=0.01,cmap='OrRd',offset=0.5,normed=False):

    def get_drop(ct,drop_dt,min_drop_ratio):
        if ct in drop_dt:
            drop = drop_dt[ct][k]
            if drop > min_drop_ratio:
                return drop
            else:
                return 0
        else:
            return 0
        
    
    cnt_dt = cdm_out["cnt"]
    drop_dt = cdm_out["drop"]
    mean_dt = cdm_out["mean"]
    
    if ctlist == None:
        ctlist = np.array(sorted(list(drop_dt.keys())))

    xpos = []
    ypos = []
    yticks = []
    yticklabels = []

    x1 = []
    y1 = []

    yidx = 0
    
    xpos_list = []
    xticks = []
    offset = 0
    
    for i,ct in enumerate(ctlist):
        if ct not in mean_dt:
            offset -= offset
        else:
            xticks.append(i+offset)
        xpos_list.append(i+offset)
    
    
    for gene in genes[::-1]:

        if gene in adata.raw.var_names:
            yidx += 1
            k = np.where(adata.raw.var_names == gene)[0][0]
            
            ymax = np.max(np.array([mean_dt[ct][k] if ct in mean_dt else 0 for ct in list(mean_dt.keys())]))

            x1.extend([get_drop(ct,drop_dt,min_drop_ratio) for ct in ctlist])
            y0 = np.array([mean_dt[ct][k] if ct in mean_dt else 0 for ct in ctlist])
            if normed:
                y1.extend(list(y0/np.max(y0)))
            else:
                y1.extend(list(y0/ymax))
            yticks.append(yidx)
            yticklabels.append(gene)
            
        else:

            yidx += 0.5
            x1.extend([0]*len(ctlist))
            y1.extend([0]*len(ctlist))
                                   
        xpos.extend(xpos_list)
        ypos.extend([yidx]*len(ctlist))

    x1 = np.array(x1)
    y1 = np.array(y1)
    
    fig = plt.figure(figsize=(len(ctlist)*figscale,len(genes)*figscale*0.9))
    ax = fig.add_subplot(1,1,1)
    sc = ax.scatter(xpos,ypos,s=x1*figscale*400,c=y1,cmap=cmap)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=fontsize)
    ax.set_xticks(xticks)
    ax.set_xticklabels([x for x in ctlist if x in mean_dt],rotation=90,fontsize=fontsize)
    ax.set_ylim(min(ypos)-0.5,max(ypos)+0.5)
    ax.set_xlim(min(xpos)-0.5,max(xpos)+0.5)
    plt.grid(False)
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sc,cax=cax)
    cbar.ax.tick_params(labelsize=8) 
    
    if save:
        plt.savefig(save,format='pdf')
    if show:
        plt.show()
    
def draw_marker_blob(adata,cdm_out,genes,ctlist = None,fontsize=10,
                     figscale=0.25,save=None, show=True,
                     min_drop_ratio=0.01,cmap='OrRd'):

    cnt_dt = cdm_out["cnt"]
    drop_dt = cdm_out["drop"]
    mean_dt = cdm_out["mean"]
    
    if ctlist == None:
        ctlist = np.array(sorted(list(drop_dt.keys())))

    xpos = []
    ypos = []
    yticks = []
    yticklabels = []

    x1 = []
    y1 = []

    yidx = 0
    
    for gene in genes[::-1]:

        if gene in adata.raw.var_names:
            yidx += 1
            k = np.where(adata.raw.var_names == gene)[0][0]

            x1.extend([drop_dt[ct][k] if ((drop_dt[ct][k] > min_drop_ratio)) else 0 for ct in ctlist])
            y0 = np.array([mean_dt[ct][k] for ct in ctlist])
            y1.extend(list(y0/np.max(y0)))
            
            yticks.append(yidx)
            yticklabels.append(gene)
            
        else:

            yidx += 0.5
            x1.extend([0]*len(ctlist))
            y1.extend([0]*len(ctlist))
                                   
        xpos.extend(list(range(len(ctlist))))
        ypos.extend([yidx]*len(ctlist))

    x1 = np.array(x1)
    y1 = np.array(y1)
    
    fig = plt.figure(figsize=(len(ctlist)*figscale,len(genes)*figscale*0.9))
    ax = fig.add_subplot(1,1,1)
    sc = ax.scatter(xpos,ypos,s=x1*figscale*400,c=y1,cmap=cmap)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=fontsize)
    ax.set_xticks(range(len(ctlist)))
    ax.set_xticklabels(ctlist,rotation=90,fontsize=fontsize)
    ax.set_ylim(min(ypos)-0.5,max(ypos)+0.5)
    ax.set_xlim(min(xpos)-0.5,max(xpos)+0.5)
    plt.grid(False)
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sc,cax=cax)
    cbar.ax.tick_params(labelsize=8) 
    
    if save:
        plt.savefig(save,format='pdf')
    if show:
        plt.show()
    

def draw_marker_blob_T(adata,cdm_out,genes,ctlist = None,fontsize=10, figscale=0.25,cmap=None,save=None):
    
    if cmap == None:
        cmap='OrRd'

    cnt_dt = cdm_out["cnt"]
    drop_dt = cdm_out["drop"]
    mean_dt = cdm_out["mean"]
    
    if ctlist == None:
        ctlist = np.array(sorted(list(drop_dt.keys())))
    totlist = np.array(sorted(list(drop_dt.keys())))

    xpos = []
    ypos = []
    xticks = []
    xticklabels = []

    x1 = []
    y1 = []

    xidx = 0
    
    ctlist = ctlist[::-1]
    
    for gene in genes:

        if gene in adata.raw.var_names:
            xidx += 1
            k = np.where(adata.raw.var_names == gene)[0][0]

            y1.extend([drop_dt[ct][k] if drop_dt[ct][k] > 0.01 else 0 for ct in ctlist])
            x0 = np.array([mean_dt[ct][k] for ct in ctlist])
            xt = np.array([mean_dt[ct][k] for ct in totlist])
            x1.extend(list(x0/np.max(x0)))
            #x1.extend(list(x0/np.max(xt)))
            
            xticks.append(xidx)
            xticklabels.append(gene)
            
        else:

            xidx += 0.5
            y1.extend([0]*len(ctlist))
            x1.extend([0]*len(ctlist))
                                   
        ypos.extend(list(range(len(ctlist))))
        xpos.extend([xidx]*len(ctlist))

    x1 = np.array(x1)
    y1 = np.array(y1)
    
    
    fig = plt.figure(figsize=(len(genes)*figscale*0.7,len(ctlist)*figscale*0.8))
    ax = fig.add_subplot(1,1,1)
    sc = ax.scatter(xpos,ypos,s=y1*figscale*400,c=x1,cmap=cmap,vmax=1.0)
    ax.xaxis.tick_top()
    ax.set_xticks(np.array(xticks))
    ax.set_xticklabels(xticklabels,rotation=90,fontsize=fontsize)
    ax.set_yticks(range(len(ctlist)))
    ax.set_yticklabels(ctlist,fontsize=fontsize)
    ax.set_ylim(min(ypos)-0.5,max(ypos)+0.5)
    ax.set_xlim(min(xpos)-0.5,max(xpos)+0.5)
    plt.grid(False)
    
    from mpl_toolkits.axes_grid1 import make_axes_locatableg
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = plt.colorbar(sc,cax=cax)
    cbar.ax.tick_params(labelsize=8) 
    if save != None:
        plt.savefig(save,bbox_inches='tight',format='pdf',dpi=300)
    plt.show()

class volcano_plot():
    '''
    creat volcano plot from anndata
    '''
    
    def __init__(self,adata,anno_key,comp1,comp2,P=0.1,quick=True):
        '''
        param P :pseudocount for fc calculation
        '''
        from scipy.stats import ttest_ind
        self.genelist = adata.raw.var_names
        index1 = adata.obs[anno_key]==comp1
        index2 = adata.obs[anno_key]==comp2

        exp1 = adata.raw[index1].X.todense()
        exp2 = adata.raw[index2].X.todense()
        
        self.pval = []
        self.fc = []
        for i in range(adata.raw.shape[1]):
            self.fc.append(np.log2((np.mean(exp1[:,i].A1)+P)/(np.mean(exp2[:,i].A1)+P)))
            if quick:
                if np.abs(self.fc[-1])<0.5:
                    self.pval.append(1)
                else:
                    self.pval.append(ttest_ind(exp1[:,i].A1,exp2[:,i].A1)[1])
            else:
                self.pval.append(ttest_ind(exp1[:,i].A1,exp2[:,i].A1)[1])
            
        self.pval = np.array(self.pval)
        self.fc = np.array(self.fc)
            
    def draw(self, pvalue_cut=100, adjust_lim = 5, show=True):
        '''
        draw volcano plot
        param pvalue_cut :-log10Pvalue for cutoff
        '''
        from adjustText import adjust_text
        plt.figure(figsize=(6,6))

        xpos = np.array(self.fc)
        ypos = -np.log10(np.array(self.pval))
        ypos[ypos==np.inf] = np.max(ypos[ypos!=np.inf])

        sig = (np.abs(xpos) > 1) & (ypos > pvalue_cut)

        plt.scatter(xpos,ypos,s=1, color='k', alpha =0.5, rasterized=True)
        plt.scatter(xpos[sig],ypos[sig],s=3,color='red', rasterized=True)

        texts = []
        for i, gene in enumerate(self.genelist[sig]):
            texts.append(plt.text(xpos[sig][i],ypos[sig][i],gene,fontsize=5))

        adjust_text(texts,only_move={'texts':'xy'},lim=adjust_lim)
        if show:
            plt.show() 
