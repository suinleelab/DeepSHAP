import pandas as pd
import numpy as np

def ablate(pred_fn, attr, x_explic, impute="pos", 
           refer=None, y_explic=None, is_loss=False):
    """
    Arguments:
    pred_fn  - prediction function for model being explained
    attr     - attribution corresponding to model and explicand
    x_explic - sample being explained
    impute   - the order in which we impute features
               "pos" imputes positive attributions by refer
               "neg" imputes negative attributions by refer
    refer    - reference to impute with (same shape as x_explic)
    y_explic - labels - only necessary for loss explanation
    is_loss  - if true, the prediction function should accept
               parameters x_explic and y_explic
    
    Returns:
    ablated_preds - predictions based on ablating
    """
    # Set reference if None
    if refer is None:
        refer = np.zeros(x_explic.shape)
    
    # Get feature rank
    if "pos" in impute: 
        feat_rank = np.argsort(-attr)
        condition = lambda x : x > 0
        
    if "neg" in impute: 
        feat_rank = np.argsort(attr)
        condition = lambda x : x < 0
        
    # Explicand to modify
    explicand = np.copy(x_explic)
    
    # Initialize ablated predictions
    if is_loss:
        assert not y_explic is None, "Need to provide labels for loss explanation"
        ablated_preds = [pred_fn(explicand, y_explic).mean()]
    else:
        ablated_preds = [pred_fn(explicand).mean()]
        
    # Ablate features one by one
    for i in range(explicand.shape[1]):
        samples   = np.arange(x_explic.shape[0])                 # Sample indices
        top_feat  = feat_rank[:,i]                               # Get top i features
        expl_val  = explicand[samples,top_feat]                  # Get values of top features
        mask_val  = expl_val                                     # Initialize mask to explicand
        cond_true = condition(attr[samples,top_feat])            # Ensure attributions satisfy positive/negative
        mask_val[cond_true] = refer[samples,top_feat][cond_true] # Update mask based on condition
        explicand[samples,top_feat] = mask_val                   # Mask top features
        
        if is_loss:
            assert not y_explic is None, "Need to provide labels for loss explanation"
            avg_pred = pred_fn(explicand, y_explic).mean()
        else:
            avg_pred = pred_fn(explicand).mean()
        ablated_preds.append(avg_pred)          # Get prediction on ablated explicand
        
    # Return ablated predictions
    return(ablated_preds)