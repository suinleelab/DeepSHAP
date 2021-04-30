import numpy as np
import os

def load_gene_set(fname):
    """
    Load gene set gmt file - assuming tab delimited with 
    gene names
    """
    tags = []; urls = []; genes = []

    with open(fname) as fp:
        line = fp.readline()

        while line:
            line = line.split("\t")

            # Process file
            tags.append(line[0])
            urls.append(line[1])
            genes.append(line[2:])

            line = fp.readline()

    return(tags, urls, genes)

def filter_max_size(tages, genes, max_size):
    """
    Filter for smaller biological processes
    Simple approach to address bias towards larger gene sets
    """
    max_size = 200
    tags2  = []; 
    genes2 = []; 
    for t, g in zip(tags, genes):
        if len(g) < max_size:
            tags2.append(t)
            genes2.append(g)
    return(tags2, genes2)

def get_beta(genes, X_train):
    """
    Get beta matrix of shape (# gene sets, # genes)
    beta_(i,j) = 1 means that gene j is in set i
    """
    beta = []
    all_gene_inds = X_train.columns.isin(genes[0])

    for i in range(len(genes)):
        gene_inds     = X_train.columns.isin(genes[i]) # Get the appropriate indices
        all_gene_inds = all_gene_inds | gene_inds      # Bookkeep covered genes
        beta.append(gene_inds)

    not_covered_gene_inds = np.invert(all_gene_inds)
    beta.append(not_covered_gene_inds)
    beta = np.array(beta)
    return(beta)

def get_pathway_attr(tags, tree_attr, beta, outcome_name, pathway_name, max_size=None):
    """
    Get pathway attributions
    """
    # Safe divide that accounts for zeros
    safe_div  = lambda a,b : np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    
    # Get gene set names and add residuals
    set_names = tags + ["RESIDUALS"]
    
    # Set the attribution path
    attr_path = "attr/{}/pathway_{}_attr.npy".format(outcome_name,"".join(pathway_name.split(".")))
    if not max_size is None:
        attr_path = "attr/{}/pathway_{}_{}_attr.npy".format(outcome_name,"".join(pathway_name.split(".")),max_size)
        
    # If pathway attributions don't exist, generate and save them
    if not os.path.exists(attr_path):
        pathway_attr = np.zeros([tree_attr.shape[0], beta.shape[0], tree_attr.shape[2]])
        for ref_ind in range(tree_attr.shape[2]):
            if ref_ind % 50 == 0: print("{} out of {}".format(ref_ind, tree_attr.shape[2]))
            attr = np.dot(tree_attr[:,:,ref_ind], beta.T)
            attr = attr * safe_div(tree_attr[:,:,ref_ind].sum(1),attr.sum(1))[:,None]
            pathway_attr[:,:,ref_ind] = attr
        np.save(attr_path, pathway_attr)
    else:
        pathway_attr = np.load(attr_path)
    assert np.allclose(pathway_attr.mean(2).sum(1),tree_attr.mean(2).sum(1))
    
    return(pathway_attr, set_names)