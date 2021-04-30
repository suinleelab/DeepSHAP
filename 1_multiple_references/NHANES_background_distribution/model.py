from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import os

from deepshap.data import load_nhanes

MLP_PATH = "models/NHANES_mlp.h5"
models = [MLP_PATH]

def nhanes_mlp(plot_convergence=False):
    """
    Train or load MLP for forecasting 15 year mortality
    """
    
    trainx, trainy, validx, validy, testx, testy = load_nhanes()
    if not os.path.exists(MLP_PATH):

        model = Sequential()
        model.add(Dense(100, input_dim=trainx.shape[1], 
                        activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        history = model.fit(trainx, trainy, epochs=50, batch_size=128,
                            validation_data=(trainx, validy))
        score = model.evaluate(testx, testy, batch_size=128)
        print("Test loss: {}, Test acc: {}".format(score[0],score[1]))
        
        if plot_convergence:
            plt.rcParams['figure.figsize'] = 6, 2
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Valid'], loc='upper left')
            plt.show()
        
        model.save("models/NHANES_mlp.h5")
    else:
        model = load_model("models/NHANES_mlp.h5")

    # Report ROC AUC
    testpred = model.predict(testx)
    print("Test ROC: {}".format(roc_auc_score(testy,testpred)))
        
    return(model)