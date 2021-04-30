from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
import os

from deepshap.data import load_gene_expr

def xgb_gene(outcome_name):
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_gene_expr(outcome_name)

    # Fit model
    mod_path = "models/{}/xgb_model.p".format(outcome_name)
    model = XGBClassifier()
    if not os.path.exists(mod_path):
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=10, verbose=False)
        model.save_model(mod_path)
    else:
        model.load_model(mod_path)

    # Evaluate
    print("Train accuracy", accuracy_score(model.predict(X_train), y_train))
    print("Valid accuracy", accuracy_score(model.predict(X_valid), y_valid))
    print("Test  accuracy", accuracy_score(model.predict(X_test), y_test))

    print("Train ROC AUC", roc_auc_score(y_train, model.predict_proba(X_train)[:,1]))
    print("Valid ROC AUC", roc_auc_score(y_valid, model.predict_proba(X_valid)[:,1]))
    print("Test  ROC AUC", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
    
    return(model)