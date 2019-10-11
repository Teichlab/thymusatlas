import numpy as np

def update_vdj(adata,vdj_df):

    AB_list = ['TRA','TRB']
    gene_key_list = ['v_gene','d_gene','j_gene']

    c_5GEX = adata.obs['method']=='5GEX'

    tcr_df = vdj_df[vdj_df.chain.isin(AB_list)] # select tcra b gene info
    gene_list = [x for x in set(list(tcr_df.v_gene)+list(tcr_df.j_gene)+list(tcr_df.d_gene)) if x !='None']

    c1 = vdj_df.full_length == True
    c2 = vdj_df.productive == 'True' # False
    c3 = vdj_df.umis > 1
    c4 = vdj_df.productive == 'False'
   
    # fl
    
    is_TRA = np.zeros(len(adata),dtype=bool)
    is_TRB = np.zeros(len(adata),dtype=bool)
    
    for gene in gene_list:
        if ('RBV' in gene) | ('RAV' in gene):
            gene_key = 'v_gene'
        elif ('RBD' in gene) | ('RAD' in gene):
            gene_key = 'd_gene'
        elif ('RBJ' in gene) | ('RAJ' in gene):
            gene_key = 'j_gene'
        else:
            print(gene)
            raise SystemError

        c_gene = vdj_df[gene_key] == gene
        c_vdj = c1&c3&c_gene #c1&c2&c3&c_gene
        vdj_list = list(vdj_df[c_vdj].obs_name)
        c_vdj = adata.obs_names.isin(vdj_list)
        
        if gene.startswith('TRA'):
            is_TRA += c_vdj
        elif gene.startswith('TRB'):
            is_TRB += c_vdj
        else:
            raise SystemError
    
    adata.obs['is_TRA_fl'] = is_TRA
    adata.obs['is_TRB_fl'] = is_TRB 
        
    # np
    
    is_TRA = np.zeros(len(adata),dtype=bool)
    is_TRB = np.zeros(len(adata),dtype=bool)
    
    for gene in gene_list:
        if ('RBV' in gene) | ('RAV' in gene):
            gene_key = 'v_gene'
        elif ('RBD' in gene) | ('RAD' in gene):
            gene_key = 'd_gene'
        elif ('RBJ' in gene) | ('RAJ' in gene):
            gene_key = 'j_gene'
        else:
            print(gene)
            raise SystemError

        c_gene = vdj_df[gene_key] == gene
        c_vdj = c1&c4&c3&c_gene #c1&c2&c3&c_gene
        vdj_list = list(vdj_df[c_vdj].obs_name)
        c_vdj = adata.obs_names.isin(vdj_list)
        
        if gene.startswith('TRA'):
            is_TRA += c_vdj
        elif gene.startswith('TRB'):
            is_TRB += c_vdj
        else:
            raise SystemError
    
    adata.obs['is_TRA_np'] = is_TRA
    adata.obs['is_TRB_np'] = is_TRB
    
    # productive
    
        
    is_TRA = np.zeros(len(adata),dtype=bool)
    is_TRB = np.zeros(len(adata),dtype=bool)
    
    for gene in gene_list:
        if ('RBV' in gene) | ('RAV' in gene):
            gene_key = 'v_gene'
        elif ('RBD' in gene) | ('RAD' in gene):
            gene_key = 'd_gene'
        elif ('RBJ' in gene) | ('RAJ' in gene):
            gene_key = 'j_gene'
        else:
            print(gene)
            raise SystemError

        c_gene = vdj_df[gene_key] == gene
        c_vdj = c1&c2&c3&c_gene #c1&c2&c3&c_gene
        vdj_list = list(vdj_df[c_vdj].obs_name)
        c_vdj = adata.obs_names.isin(vdj_list)
        adata.obs['VDJ_'+gene] = c_vdj
        
        if gene.startswith('TRA'):
            is_TRA += c_vdj
        elif gene.startswith('TRB'):
            is_TRB += c_vdj
        else:
            raise SystemError
    
    adata.obs['is_TRA_p'] = is_TRA
    adata.obs['is_TRB_p'] = is_TRB