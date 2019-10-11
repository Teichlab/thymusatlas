import numpy as np
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

def draw_3d(adata,group_key,version,size = 2):

    color_list = adata.uns[group_key+'_colors']
    trace_list = []

    for i, ct in enumerate(set(adata.obs[group_key])):
        sdata = adata[adata.obs[group_key]==ct]

        x = sdata.obsm['X_umap'][:,0]
        y = sdata.obsm['X_umap'][:,1]
        z = sdata.obsm['X_umap'][:,2]

        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            name=ct,
            mode='markers',
            marker=dict(
                size=size,
                color=color_list[i],
                opacity=0.8
            )
        )

        trace_list.append(trace)

    data = trace_list
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='%s-%s-umap-3d-scatter.html'%(version,group_key))
    
def draw_dotplot_scale(fig_folder):
    figscale= 1.5
    fig = plt.figure(figsize=(3,2))
    ax = fig.add_subplot(1,1,1)
    xpos = [1,1.27,1.61,2]
    x1 = np.array([0.25,0.5,0.75,1])
    ypos = [1,1,1,1]
    ax.scatter(xpos,ypos,s=x1*figscale*400,c='grey',cmap='OrRd',marker='o')
    ax.set_xlim(0.5,2.5)
    plt.grid(False)
    plt.savefig(fig_folder+'Dopplot.scale.pdf',format='pdf')
    
    
def draw_pseudo_heatmap(adata,genelist):
    
    import sklearn
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    
    Xt = adata.raw.X
    Rt = Xt.expm1()
    t = adata.obs['dpt_order']
    Rts = Rt[np.argsort(np.array(t))]
    bins = np.linspace(0,len(t),20)
    binlabel = np.digitize(np.array(t), bins)
    adata.obs['dpt_bins'] = [str(x) for x in binlabel]

    gidx = adata.raw.var_names.isin(genelist)
    gene_order = list(adata.raw.var_names[gidx])
    re_order = [gene_order.index(x) for x in genelist]

    normed = sklearn.preprocessing.normalize(Rts[:,gidx][:,re_order].todense().T,norm='max')
    runned = np.apply_along_axis(lambda x: running_mean(x,100),1,normed)
    normed = sklearn.preprocessing.normalize(runned,norm='max')

    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)

    height=12
    ax.imshow(normed, extent = [0,10,0,height], vmax=1,origin='upper',cmap='RdYlBu_r')
    ax.set_yticks(np.linspace(0,height,len(genelist)+1)[:-1]+height/(len(genelist)*2))
    ax.set_yticklabels(genelist[::-1],fontsize=6)
    ax.set_xticks([])
    ax.grid(False)
    #ax.set_aspect(0.1)
    plt.show()

def draw_pseudo_heatmap_anno(adata,anno,color_list =None,anno_list=None):
    import sklearn
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import colors

    t = adata.obs['dpt_order']
    anl = list(set(adata.obs[anno]))
    Ob = np.vstack([np.array(adata.obs[anno]==an) for an in anl]).astype(int).T
    Obs = Ob[np.argsort(np.array(t))]
    
    order = anl
    if anno_list!=None:
        re_order = [anl.index(x) for x in anno_list]
    else:
        re_order = [anl.index(x) for x in anl]
        anno_list = anl

    normed = sklearn.preprocessing.normalize(Obs[:,re_order].T,norm='max')
    runned = np.apply_along_axis(lambda x: running_mean(x,1),1,normed)
    normed = sklearn.preprocessing.normalize(runned,norm='max')

    fig = plt.figure(figsize=(3,0.5))
    ax = fig.add_subplot(111)

    height=1.4

    clist = ['white', 'red','green','blue','orange','skyblue','cyan','magenta']
    clist = ['white']+adata.uns[anno+'_colors']
    if clist:
        clist = ['white']+color_list

    for i in range(len(anl)):
        print(i, height/len(anl)*i,height/len(anl)*(i+1), anno_list[i])
        cmap = colors.ListedColormap([clist[0],clist[i+1]])
        ax.imshow(normed[i:i+1], extent = [0,10,height/(len(anl))*i,height/(len(anl))*(i+1)], vmax=1,origin='upper',cmap=cmap)
    ax.set_yticks(np.linspace(0,height,len(anno_list)+1)[:-1]+height/(len(anno_list)*2))
    ax.set_yticklabels(anno_list,fontsize=8)
    ax.set_xticks([])
    ax.set_ylim(0,height)
    ax.grid(False)
    #ax.set_aspect(0.1)
    plt.show()
    

def draw_pseudo_heatmap_v2(adata,genelist,anno=None, clist=None, anno_list =None, gridspec_kw = {'height_ratios': [10,1]},figsize=(3,4),fontsize=12, mean_window = 100, anno_window=10):
    
    import sklearn
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import colors
    
    
    fig,(axes) = plt.subplots(2, 1,  sharex= True,
                        gridspec_kw=gridspec_kw,
                        figsize=figsize)
    
    
    Xt = adata.raw.X
    Rt = Xt.expm1()
    t = adata.obs['dpt_order']
    Rts = Rt[np.argsort(np.array(t))]
    bins = np.linspace(0,len(t),20)
    binlabel = np.digitize(np.array(t), bins)
    adata.obs['dpt_bins'] = [str(x) for x in binlabel]

    gidx = adata.raw.var_names.isin(genelist)
    gene_order = list(adata.raw.var_names[gidx])
    re_order = [gene_order.index(x) for x in genelist]

    normed = sklearn.preprocessing.normalize(Rts[:,gidx][:,re_order].todense().T,norm='max')
    runned = np.apply_along_axis(lambda x: running_mean(x,mean_window),1,normed)
    normed = sklearn.preprocessing.normalize(runned,norm='max')
    

    ax = axes[0]

    height=12
    im = ax.imshow(normed, extent = [0,10,0,height], vmax=1,origin='upper',cmap='RdYlBu_r')
    ax.set_yticks(np.linspace(0,height,len(genelist)+1)[:-1]+height/(len(genelist)*2))
    ax.set_yticklabels(genelist[::-1],fontsize=fontsize)
    ax.set_xticks([])
    ax.grid(False)
    ax.set_aspect('auto')
    
    if anno:

        anl = sorted(list(set(adata.obs[anno])))
        Ob = np.vstack([np.array(adata.obs[anno]==an) for an in anl]).astype(int).T
        Obs = Ob[np.argsort(np.array(t))]

        order = anl
        if anno_list!=None:
            re_order = [anl.index(x) for x in anno_list]
        else:
            re_order = [anl.index(x) for x in anl]
            anno_list = anl

        normed = sklearn.preprocessing.normalize(Obs[:,re_order].T,norm='max')
        runned = np.apply_along_axis(lambda x: running_mean(x,anno_window),1,normed)
        normed = sklearn.preprocessing.normalize(runned,norm='max')
        normed = normed>0
        ax = axes[1]
        
        height=1.4


        if clist:
            clist = ['white']+clist
        else:
            clist = ['white']+list(np.array(adata.uns[anno+"_colors"])[re_order])

        for i in range(len(anl)):
            print(i, height/len(anl)*i,height/len(anl)*(i+1), anno_list[i])
            cmap = colors.ListedColormap([clist[0],clist[i+1]])
            ax.imshow(normed[i:i+1], extent = [0,10,height/(len(anl))*i,height/(len(anl))*(i+1)], vmax=1,origin='upper',cmap=cmap)
        ax.set_yticks(np.linspace(0,height,len(anno_list)+1)[:-1]+height/(len(anno_list)*2))
        ax.set_yticklabels(anno_list,fontsize=fontsize)
        ax.set_xticks([])
        ax.set_ylim(0,height)
        ax.grid(False)
        ax.set_aspect('auto')

    plt.subplots_adjust(wspace=0, hspace=0.01)
    
    return im