import numpy as np
import pickle
import os

def model_pred(mod, X):
    """
    Model prediction (for XGBClassifier and keras MLP)
    """
    if "XGB" in str(type(mod)):
        pred = mod.predict_proba(X)[:,1]
    elif "GradientBoostingClassifier" in str(type(mod)):
        pred = mod.predict_proba(X)[:,1]
    elif "tensorflow" in str(type(mod)):
        pred = mod.predict(X)[:,0]*.999
    return(pred)

def models_pred(models, X):
    """
    Params
    ======
    models : List of models (Currently accepts XGBClassifier 
             and Keras MLP)
    X : X data to make predictions on
    
    Returns
    =======
    pred_arr : Returns predictions with shape (# samples in X, # models)
    """
    pred_arr = np.vstack([model_pred(mod, X) for mod in models]).T
    return(pred_arr)

#####################################
### Gradient boosting trees (XGB) ###
#####################################
from xgboost import XGBClassifier

def fit_xgb_trval(X_train, y_train, X_valid=None, y_valid=None):
    """
    Fit xgboost model for available data.  Depending on if validation
    data is provided, use an evaluation set.
    """
    
    xgb_model = XGBClassifier()
    if not X_valid is None and not y_valid is None:
        xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], 
                        early_stopping_rounds=10, verbose=False)
    else:
        xgb_model.fit(X_train, y_train, verbose=False)
    return(xgb_model)

def fit_xgb(X_train, y_train, X_valid=None, y_valid=None, mname=None):
    """
    Params
    ======
    X_train : Train X data
    y_train : Train y data
    X_valid : Validation X data
    y_valid : Validation y data
    mname   : The name of the model (for saving in models/ dir)
    
    Returns
    =======
    xgb_model : Return the trained model
    """
    
    # Don't save or try to load if no model name
    if mname is None:
        xgb_model = fit_xgb_trval(X_train, y_train, X_valid, y_valid)
        return(xgb_model)
    
    # Train or load model depending on if it exists
    mpath = "models/"+mname
    if not os.path.exists(mpath):
        xgb_model = fit_xgb_trval(X_train, y_train, X_valid, y_valid)
        pickle.dump(xgb_model, open(mpath,"wb"))
    else:
        xgb_model = pickle.load(open(mpath,"rb"))
    
    return(xgb_model)

#########################################
### Gradient boosting trees (sklearn) ###
#########################################
from sklearn.ensemble import GradientBoostingClassifier

def fit_gbc(X_train, y_train, X_valid=None, y_valid=None, mname=None):
    """
    Params
    ======
    X_train : Train X data
    y_train : Train y data
    X_valid : Validation X data
    y_valid : Validation y data
    mname   : The name of the model (for saving in models/ dir)
    
    Returns
    =======
    xgb_model : Return the trained model
    """
    
    # Don't save or try to load if no model name
    if mname is None:
        gbc_model = GradientBoostingClassifier()
        gbc_model.fit(X_train, y_train)
        return(gbc_model)
    
    # Train or load model depending on if it exists
    mpath = "models/"+mname
    if not os.path.exists(mpath):
        gbc_model = GradientBoostingClassifier()
        gbc_model.fit(X_train, y_train)
        pickle.dump(gbc_model, open(mpath,"wb"))
    else:
        gbc_model = pickle.load(open(mpath,"rb"))
    
    return(gbc_model)

#####################################
### Multi-layer perceptrons (MLP) ###
#####################################
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input

def fit_mlp(X_train, y_train, X_valid=None, y_valid=None, mname=None, eta=0.001):
    """
    Params
    ======
    X_train : Train X data
    y_train : Train y data
    X_valid : Validation X data
    y_valid : Validation y data
    mname   : The name of the model (for saving in models/ dir)
    
    Returns
    =======
    mlp_model : Return the trained model
    """
    # Don't save or try to load if no model name
    if mname is None:
        mlp_model = Sequential()
        mlp_model.add(Dense(100, activation='relu', input_shape=(X_train.shape[1],)))
        mlp_model.add(Dense(100, activation='relu'))
        mlp_model.add(Dense(1, activation='sigmoid'))

        mlp_model.compile(loss='binary_crossentropy',optimizer=SGD(learning_rate=eta))
#         mlp_model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.01))
        mlp_model.fit(X_train,y_train,epochs=10,verbose=False)
        return(mlp_model)
    
    # Train or load model depending on if it exists
    mpath = "models/" + mname
    if not os.path.exists(mpath):
        mlp_model = Sequential()
        mlp_model.add(Dense(100, activation='relu', input_shape=(X_train.shape[1],)))
        mlp_model.add(Dense(100, activation='relu'))
        mlp_model.add(Dense(1, activation='sigmoid'))

        mlp_model.compile(loss='binary_crossentropy',optimizer=SGD(learning_rate=eta))
#         mlp_model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.01))
        mlp_model.fit(X_train,y_train,epochs=10,verbose=False)
        mlp_model.save(mpath)
    else:
        mlp_model = load_model(mpath)
    
    return(mlp_model)