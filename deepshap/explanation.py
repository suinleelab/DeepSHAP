from lime.lime_tabular import LimeTabularExplainer as LE
import numpy as np

##########################################
###### Wrapper for LIME explanation ######
##########################################
def lime_wrapper(model, explicand, reference, mode="regression", 
                 wrap_pred=True, num_samples=5000):
    """
    Get LIME explanations
    
    Args
     - model : model to explain
     - explicand : samples to explain
     - reference : samples to compare to
     - mode : whether the model is regression or classification
     - wrap_pred : whether we need to wrap the prediction function
     - num_samples : number of samples to evaluate for lime
     
    Returns
     - attr : numpy array with the lime attributions
    """
    if "DataFrame" in str(type(reference)):
        reference = reference.values
    explainer = LE(reference, verbose=False, mode=mode)

    if wrap_pred:
        def pred_wrapper(x):
            return(model.predict(x))
    else:
        pred_wrapper = model # The model is a prediction function

    attr_lst = []
    for j in range(explicand.shape[0]):
        expXGB = explainer.explain_instance(explicand[j], pred_wrapper, 
                                            num_features=reference.shape[1],
                                            num_samples=num_samples)
        attr = np.zeros(reference.shape[1])
        for feat, attr_val in expXGB.local_exp[1]:
            attr[feat] = attr_val
        attr_lst.append(attr)
    return(np.array(attr_lst))

#################################################
###### Partial dependence plot explanation ######
#################################################
def pdp(pred_fn, explicand, reference):
    """
    Get partial dependence plot explanations
    
    Args
     - pred_fn : model's prediction function
     - explicand : the samples being explained
     - reference : the value to impute with
    
    Returns
     - attr_lst : the final attributions
    """
    attr_lst = []
    for feat in range(explicand.shape[1]):
        explicand2 = np.copy(explicand)
        for i in range(explicand.shape[1]):
            if i == feat: continue
            explicand2[:,i] = reference[0,i]
        attr_lst.append(pred_fn(explicand2))
    np.array(attr_lst)
    return(np.array(attr_lst).T)

############################################
###### DeepSHAP for stack of nn->tree ######
############################################
import numpy as np
import xgboost
import sys

# Import local version of shap
sys.path.insert(0, "../../shap")
import shap

class nntree(object):
    """
    Stack of neural network to tree model
    """
    def __init__(self, nn_model, tree_model):
        self.nn_model   = nn_model
        self.tree_model = tree_model

    def fit(self, X, y, verbose=0):
        pass

    def predict(self, X, output_margin=True):
        if "pandas" in str(type(X)): 
            X = X.values
        X_embed = xgboost.DMatrix(self.nn_model.predict(X))
        return(self.tree_model.predict(X_embed, output_margin=output_margin))

def stacked_shap_nn_tree(model, explicand, reference):
    """
    Stacked SHAP to explain a model composed of a neural 
    network feature extractor fed into a tree model
    
    Args
     - model : nntree object that contains a neural network 
               and tree model
     - explicand : the sample to explain
     - reference : the sample to compare to
     
    Returns
     - se_attr : the stacked shap attributions based on
                 the rescale rule
    """
    safe_div = lambda n, d : np.divide(n, d, out=np.zeros_like(n), where=d!=0) 
    
    # Get embedded versions of reference/explicand
    reference_embed = model.nn_model.predict(reference)
    explicand_embed = model.nn_model.predict(explicand)

    # Set up explainer objects
    t_expl = shap.TreeExplainer(model.tree_model, reference_embed, feature_dependence="interventional")
    d_expl = shap.DeepExplainer(model.nn_model, reference)

    # Obtain single reference shap values for tree (#foreground, #hidden, #background)
    it_attr_pr = t_expl.shap_values(explicand_embed, model_stack=True)
    it_attr = it_attr_pr.mean(2)

    # Rescale shap values to get gradient
    num_references = reference_embed.shape[0]
    num_explicands = explicand_embed.shape[0]

    explicand_embed2 = np.transpose(explicand_embed)[np.newaxis,:,:]
    explicand_embed2 = np.repeat(explicand_embed2,num_references,axis=0)
    reference_embed2 = reference_embed[:,:,np.newaxis]
    reference_embed2 = np.repeat(reference_embed2,num_explicands,axis=2)

    # (#background, #hidden, #foreground)
    numer    = np.transpose(it_attr_pr)
    denom    = explicand_embed2 - reference_embed2
    rescale_factor = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) 

    # Make sure stacked attributions sum up correctly
    de_attr_pr = np.array(d_expl.shap_values(explicand, model_stack=True))
    de_attr    = de_attr_pr.mean(2).sum(0)

    # Reshape rescale_factor
    rescale_factor2 = np.swapaxes(rescale_factor, 0, 1)
    rescale_factor2 = np.swapaxes(rescale_factor2, 1, 2)
    assert rescale_factor2.shape == de_attr_pr.shape[:3]

    # Rescale
    if len(de_attr_pr.shape) == 6:
        se_attr_pr = rescale_factor2[:,:,:,None,None,None] * de_attr_pr
    elif len(de_attr_pr.shape) == 4:
        se_attr_pr = rescale_factor2[:,:,:,None] * de_attr_pr
    else:
        se_attr_pr = np.copy(de_attr_pr)
        for i in range(de_attr_pr.shape[0]):
            for j in range(de_attr_pr.shape[1]):
                for k in range(de_attr_pr.shape[2]):
                    se_attr_pr[i,j,k,:] = rescale_factor2[i,j,k]*de_attr_pr[i,j,k,:]

    # Take mean across references and sum across hidden outputs
    se_attr = se_attr_pr.mean(2).sum(0)

    # Check for efficiency (everything sums up correctly)
    se_attr_sum = se_attr.sum(axis=tuple(range(1, se_attr.ndim)))
    it_attr_sum = it_attr.sum(1)
    assert np.allclose(se_attr_sum, it_attr_sum, atol=1e-5)
    
    return(se_attr)

############################################
###### Integrated Gradients for Keras ######
############################################
import keras.backend as K
import pandas as pd
import numpy as np
import shap, xgboost

def lin_interp(sample, reference=False, num_steps=100):
    """
    Get linear interpolations between reference and sample.
    """
    # Use default reference values if reference is not specified
    if reference is False: reference = np.zeros(sample.shape);

    # Reference and sample shape needs to match exactly
    assert sample.shape == reference.shape

    # Calcuated stepwise difference from reference to the actual sample.
    ret = np.zeros(tuple([num_steps+1] + [i for i in sample.shape]))
    for s in range(num_steps+1):
        ret[s] = reference+(sample-reference)*(float(s)/num_steps)

    return(ret)

def int_grad(model, explicand, reference, num_steps=1000):
    """
    Integrated gradients implementation for Keras.
    Modified version of https://github.com/hiranumn/IntegratedGradients
    
    Args
     - model : model being explained
     - explicand : sample we interpolate to
     - reference : sample we interpolate from
     - num_steps : number of interpolations
    
    Returns
     - attr : integrated gradients feature attributions 
    """
    attr = []
    for i in range(explicand.shape[0]):
        explicand_curr = np.array(explicand)[i:i+1,:]
        reference_curr = np.array(reference)
        
        # Get function that returns gradients of the output w.r.t. inputs
        gradients = model.optimizer.get_gradients(model.output, model.input)
        get_grad  = K.function(inputs=[model.input], outputs=gradients)

        # Get many interpolations of explicand with zero baseline
        interp_arr = lin_interp(explicand_curr, reference_curr, num_steps=num_steps)

        # Get gradients and take the mean
        gradients_arr = [get_grad([interp_arr[i]]) for i in range(num_steps)]
        gradients_arr = np.array(gradients_arr)
        integrate_arr = gradients_arr.mean(0)[0,:][0,:]
        
        # Multiply by input minus baseline
        attr.append((explicand_curr[0,:]-reference_curr[0,:])*integrate_arr)
    attr = np.array(attr)
    
    return(attr)

########################################################
###### Generalized Integrated Gradients for Trees ######
########################################################
DEBUG = False

def find_thresholds(trees, nfeats):
    """
    Step 1 - Find all thresholds
    """    
    feat_thresholds = [[] for i in range(nfeats)]
    all_thresholds  = []
    for tree_ind in range(trees.features.shape[0]):
        for node_ind in range(trees.features.shape[1]):
            ind = (tree_ind,node_ind)

            # Leaf node
            if trees.children_left[ind] == -1: continue
            # Get feature and threshold
            feat = trees.features[ind]; thres = trees.thresholds[ind]
            # Check for valid features
            if feat == -1: continue
            # Append to lists
            feat_thresholds[feat].append(thres)
            all_thresholds.append(thres)
    return(feat_thresholds, all_thresholds)

def find_alphas(explicand, reference, feat_thresholds):
    """
    Step 2 - Find alphas
    """    
    expl = np.array(explicand)
    refe = np.array(reference)
    alpha_feats = []
    for feat in range(explicand.shape[1]):
        for thres in np.unique(feat_thresholds[feat])  :
            a = expl[0,feat]; b = refe[0,feat]
            if min(a,b) < thres and thres < max(a,b):
                den = max(a,b) - min(a,b)
                num = np.abs(thres - b)
                alpha = num/den
                alpha_feats.append((alpha,feat))
    return(alpha_feats)

def get_phi(model, explicand, reference, all_thresholds, alpha_feats):
    """
    Step 3 - Make Perturbations
    """    
    # Figure out how much to perturb by
    sort_thres = np.array(sorted(all_thresholds))
    thres_diff = sort_thres[1:] - sort_thres[:-1]
    min_diff   = min(thres_diff[thres_diff != 0])
    perturb    = min_diff/2.0

    gig_attr    = np.zeros(explicand.shape[1])
    alpha_feats = sorted(alpha_feats, key=lambda x : x[0])
    check_preds = [model.predict(xgboost.DMatrix(reference))]
    for (alpha,feat) in alpha_feats:
        # Get predictions on left and right side of interpolation
        xpert  = explicand - reference
        xleft  = (alpha * xpert) - (np.sign(xpert) * perturb)
        xright = (alpha * xpert) + (np.sign(xpert) * perturb)
        pred_left  = model.predict(xgboost.DMatrix(xleft))[0]
        pred_right = model.predict(xgboost.DMatrix(xright))[0]

        # Give the appropriate feature all the credit
        gig_attr[feat] += pred_right - pred_left

        # Collect predictions along path for debugging
        if pred_left != pred_right: check_preds.append(pred_left)
    check_preds.append(pred_right)
    if DEBUG: print(check_preds)
    return(gig_attr)

def tree_gig_one_sample(model, explicand, reference):
    """
    Generalized integrated gradients implementation for tree models.
    
    Args
     - model     : model being explained
     - explicand : sample we interpolate to
     - reference : sample we interpolate from
    
    Returns
     - attr : generalized integrated gradients feature attributions 
    """
    # Rely on SHAP to process trees
    explainer = shap.TreeExplainer(model, reference, feature_dependence="independent")
    trees = explainer.model

    # Process trees for thresholds
    feat_thresholds, all_thresholds = find_thresholds(trees, explicand.shape[1])
    
    # Get interpolations to consider
    alpha_feats = find_alphas(explicand, reference, feat_thresholds)
    
    # Get attributions
    attr = get_phi(model, explicand, reference, all_thresholds, alpha_feats)

    return(attr)

def tree_gig(model, explicand, reference):
    """
    Generalized integrated gradients implementation for tree models.
    
    Wrapper that supports multiple explicands.
    """
    
    explicand = np.array(explicand)
    attr = []
    for i in range(explicand.shape[0]):
        attr.append(tree_gig_one_sample(model, explicand[i:i+1,:], reference))

    return(np.array(attr))
