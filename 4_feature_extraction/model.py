import os, pickle 
import xgboost as xgb
import numpy as np

from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

# Add parent directory to sys path for imports
from deepshap.data import load_mnist
from deepshap.explanation import nntree

#########################
### Model evaluations ###
#########################
# Get accuracies
def get_acc(model, data, labels):
    pred = model.predict(data)
    if len(pred.shape) == 1:
        pred = pred > 0.5
    accu = metrics.accuracy_score(labels, pred)
    return(accu)

# Get accuracies for 95% CI
def bootstrap_test(labels, preds):
    acc_lst = []
    for j in range(1000):
        inds = np.random.randint(0,labels.shape[0],labels.shape[0])
        acc  = metrics.accuracy_score(labels[inds], preds[inds])
        acc_lst.append(acc)
    return(acc_lst)


##############
### Models ###
##############
input_shape = (28, 28, 1)
num_classes = 10

CNN2PATH   = "models/mnist_cnn2.h5"
EMBXGBPATH = "models/mnist_embed_xgb.p"

def mnist_cnn2():
    """
    CNN for MNIST - Using dense layer after flatten so we can 
    use DeepSHAP
    """
    
    x_train, y_train, x_test, y_test = load_mnist(binary_ind=None, argmax=False)

    # Train model if not saved
    if not os.path.exists(CNN2PATH):
        model = keras.Sequential(
            [
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu", 
                              input_shape=input_shape),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(100, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(loss="categorical_crossentropy", 
                      optimizer="adam", 
                      metrics=["accuracy"])
        
        model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)
        model.save(CNN2PATH)
    else:
        model = models.load_model(CNN2PATH)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    return(model)

XGB_PARAM = {
    'max_depth': 5,                 # the maximum depth of each tree
    'eta': 0.5,                     # the training step for each iteration
    'objective': 'multi:softmax',   # multiclass classification using the softmax objective
    'num_class': 10,                # the number of classes that exist in this datset
    'early_stopping_rounds': 10,
}

XGB_PARAM_BIN = {
    'max_depth': 5,                 # the maximum depth of each tree
    'eta': 0.5,                     # the training step for each iteration
    'objective': 'binary:logistic', # multiclass classification using the softmax objective
    'early_stopping_rounds': 10,
}

def mnist_cnn_xgb(num_trees=50, binary_ind=None):
    """ 
    CNN to XGB for corrgroups60
    """
    # Load data
    x_train, y_train, x_test, y_test = load_mnist(binary_ind=binary_ind)
    
    # XGB model path
    EMBXGBPATH = "models/mnist_embed_xgb{}.p".format(num_trees)
    
    ### Train or load CNN ###
    cnn_mod = mnist_cnn2()
    
#     emb_mod = keras.models.Sequential()
#     for layer in cnn_mod.layers[:-1]: # Exclude last layer
#         emb_mod.add(layer)
      
    # Create model and load weights manually for deepshap compatibility
    emb_mod = keras.Sequential(
        [
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape,
                          weights=cnn_mod.layers[0].get_weights()),
            layers.MaxPooling2D(pool_size=(2, 2), weights=cnn_mod.layers[1].get_weights()),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu",
                          weights=cnn_mod.layers[2].get_weights()),
            layers.MaxPooling2D(pool_size=(2, 2), weights=cnn_mod.layers[3].get_weights()),
            layers.Flatten(weights=cnn_mod.layers[4].get_weights()),
            layers.Dense(100, activation="relu", weights=cnn_mod.layers[5].get_weights())
        ]
    )
    
    ### Train or load XGB ###
    emb_train_data = xgb.DMatrix(emb_mod.predict(x_train), label=y_train)
    if not os.path.exists(EMBXGBPATH):
        if binary_ind is None:
            xgb_model = xgb.train(XGB_PARAM, emb_train_data, num_trees)
        else:
            xgb_model = xgb.train(XGB_PARAM_BIN, emb_train_data, num_trees)
        pickle.dump(xgb_model, open(EMBXGBPATH, "wb"))
    else:
        xgb_model = pickle.load(open(EMBXGBPATH, "rb"))
    
    ### Create stacked model ###
    model = nntree(emb_mod, xgb_model)
    
    # Print the accuracy
    train_accu = get_acc(xgb_model, xgb.DMatrix(emb_mod.predict(x_train)), y_train)
    test_accu  = get_acc(xgb_model, xgb.DMatrix(emb_mod.predict(x_test)), y_test)
    print("Train Accuracy:", train_accu)
    print("Test Accuracy: ", test_accu)
    
    return(model)