import pandas as pd
from adjustText import adjust_text
import numpy as np
import matplotlib.pyplot as plt
from scjp import jp_save_fig

hom = pd.read_table('D03_HOM_MouseHumanSequence.rpt.txt')

c_m = hom['NCBI Taxon ID']==10090
c_h = hom['NCBI Taxon ID']==9606

MtH = {}
HtM = {}
for HomID in sorted(set(hom['HomoloGene ID'])):
    c1 = hom['HomoloGene ID']==HomID
    try: hum = np.array(hom[c1&c_h]['Symbol'])[0]
    except: continue
    try: mus = np.array(hom[c1&c_m]['Symbol'])[0]
    except: continue
    MtH[mus] = hum 
    HtM[hum] = mus
    
def mouse_to_human(x):
    if x in MtH:
        return MtH[x]
    else:
        return x.upper()
    

def draw_scatter_speices_ct(ct,fc,version,fig_folder,percentile_thres=99.5,
                           adjust_lim = 3,fontsize=4):
    v3 = fc[ct]['hs']
    v4 = fc[ct]['mm']

    xpos = np.array(v3.fc) 
    ypos = np.array(v4.fc)

    ymax = np.percentile(ypos,percentile_thres)
    xmax = np.percentile(xpos,percentile_thres)

    sig = (((ypos) > ymax) & (-np.log10(v4.pval) > 5))|\
           (((xpos) > xmax) & (-np.log10(v3.pval) > 5))

    plt.scatter(xpos,ypos,s=1, color='k', alpha =0.5, rasterized=True)
    plt.title(ct)
    texts = []
    for i, gene in enumerate(v3.genelist[sig]):
        texts.append(plt.text(xpos[sig][i],ypos[sig][i],gene,fontsize=fontsize))
    if adjust_lim:
        adjust_text(texts,only_move={'texts':'xy'},lim=adjust_lim)
    plt.grid(False)
    jp_save_fig(version,'FigS1_epi_%s'%ct,fig_folder=fig_folder)
    
    
def draw_scatter_speices_ct_v2(ct,fc,version,fig_folder,percentile_thres=99.5,
                           adjust_lim = 3, fontsize=5):
    v3 = fc[ct]['h_m']
    v4 = fc[ct]['mks']

    xpos = np.array(v3.fc) 
    ypos = np.array(v4.fc)

    ymax = np.percentile(ypos,percentile_thres)
    xmax = np.percentile(np.abs(xpos),percentile_thres)
    
    sig1 = (((ypos) > ymax) & (-np.log10(v4.pval) > 5)) 
    sig2 = ((np.abs(xpos) > xmax) & (-np.log10(v3.pval) > 5))

    sig = sig1
           

    plt.scatter(xpos,ypos,s=1, color='k', alpha =0.5, rasterized=True)
    plt.title(ct)
    texts = []
    for i, gene in enumerate(v3.genelist[sig]):
        texts.append(plt.text(xpos[sig][i],ypos[sig][i],gene,fontsize=fontsize))
    if adjust_lim:
        adjust_text(texts,only_move={'texts':'xy'},lim=adjust_lim)
    plt.grid(False)
    jp_save_fig(version,'FigS1_species_v2_%s'%ct,fig_folder=fig_folder)