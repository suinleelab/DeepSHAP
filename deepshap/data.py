##################################
### MNIST digit classification ###
##################################
import numpy as np
from tensorflow import keras

def load_mnist(verbose=False, binary_ind=None, argmax=True):
    """
    Params
    ======
    
    verbose : Whether to print size of train/test
    binary_ind : Return a single class or all ten
    argmax : If binary_ind is false, should we take argmax of 10 classes
    """
    
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # Split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    if verbose:
        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    # Convert outcome to binary
    if binary_ind is None: 
        if argmax:
            y_train = np.argmax(y_train,1)
            y_test  = np.argmax(y_test,1)
    else:
        y_train = y_train[:,binary_ind]
        y_test  = y_test[:,binary_ind]
    
    return(x_train, y_train, x_test, y_test)




#####################################
### NHANES I mortality prediction ###
#####################################
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
from deepshap import loadnhanes

name_map = {
    "sex_isFemale": "Sex",
    "age": "Age",
    "systolic_blood_pressure": "Systolic blood pressure",
    "weight": "Weight",
    "height": "Height",
    "white_blood_cells": "White blood cells", # (mg/dL)
    "sedimentation_rate": "Sedimentation rate",
    "serum_albumin": "Blood albumin",
    "alkaline_phosphatase": "Alkaline phosphatase",
    "cholesterol": "Total cholesterol",
    "physical_activity": "Physical activity",
    "hematocrit": "Hematocrit",
    "uric_acid": "Uric acid",
    "red_blood_cells": "Red blood cells",
    "urine_albumin_isNegative": "Albumin present in urine",
    "serum_protein": "Blood protein"
}

def preprocessed_data():
    """
    Return the preprocessed X and y
    """
    X,y = loadnhanes._load()

    # clean up a bit
    for c in X.columns:
        if c.endswith("_isBlank"):
            del X[c]
    del X["urine_hematest_isTrace"] # would have no variance in the strain set
    del X["SGOT_isBlankbutapplicable"] # would have no variance in the strain set
    del X["calcium_isBlankbutapplicable"] # would have no variance in the strain set
    del X["uric_acid_isBlankbutapplicable"] # would only have one true value in the train set
    del X["urine_hematest_isVerylarge"] # would only have one true value in the train set
    del X["total_bilirubin_isBlankbutapplicable"] # would only have one true value in the train set
    del X["alkaline_phosphatase_isBlankbutapplicable"] # would only have one true value in the train set
    del X["hemoglobin_isUnacceptable"] # redundant with hematocrit_isUnacceptable
    nan_rows = np.isnan(X["height"]) | np.isnan(X["weight"]) | np.isnan(X["systolic_blood_pressure"])
    rows = np.where(np.invert(nan_rows))[0]
    X = X.iloc[rows,:]
    y = y[rows]
    # Convert to binary prediction of mortality
    y = y > 0 
    
    return(X,y)
    
def get_nhanes_feat_names():
    """
    Get the appropriate feature names (for SHAP plots)
    """
    X, _ = preprocessed_data()
    mapped_feature_names = list(map(lambda x: name_map.get(x, x), X.columns))
    return(mapped_feature_names)

def load_nhanes():
    """
    Load nhanes I data.  
    Pre-processed data with imputation.
    """
    X,y = preprocessed_data()

    # split by patient id
    pids = np.unique(X.index.values)
    train_pids,test_pids = train_test_split(pids, random_state=0)
    strain_pids,valid_pids = train_test_split(train_pids, random_state=0)

    # find the indexes of the samples from the patient ids
    train_inds = np.where([p in train_pids for p in X.index.values])[0]
    strain_inds = np.where([p in strain_pids for p in X.index.values])[0]
    valid_inds = np.where([p in valid_pids for p in X.index.values])[0]
    test_inds = np.where([p in test_pids for p in X.index.values])[0]

    # create the split datasets
    X_train = X.iloc[train_inds,:]
    X_strain = X.iloc[strain_inds,:]
    X_valid = X.iloc[valid_inds,:]
    X_test = X.iloc[test_inds,:]
    y_train = y[train_inds]
    y_strain = y[strain_inds]
    y_valid = y[valid_inds]
    y_test = y[test_inds]

    # mean impute for linear and deep models
    imp = SimpleImputer()
    imp.fit(X_strain)
    X_strain_imp = imp.transform(X_strain)
    X_train_imp = imp.transform(X_train)
    X_valid_imp = imp.transform(X_valid)
    X_test_imp = imp.transform(X_test)
    X_imp = imp.transform(X)

    # standardize
    scaler = StandardScaler()
    scaler.fit(X_strain_imp)
    X_strain_imp = scaler.transform(X_strain_imp)
    X_train_imp = scaler.transform(X_train_imp)
    X_valid_imp = scaler.transform(X_valid_imp)
    X_test_imp = scaler.transform(X_test_imp)
    X_imp = scaler.transform(X_imp)

    return(X_strain_imp, y_strain, X_valid_imp, y_valid, X_test_imp, y_test)



#############################################
### NHANES 1999-2014 mortality prediction ###
#############################################
from sklearn.model_selection import train_test_split
import pandas as pd

# Extra weight features to drop from variables
extra_weight_feats = ["Questionnaire_TriedToLoseWeight", 
                      "Questionnaire_SelfReportedGreatestWeight",
                      "Questionnaire_SelfReportedWeight", 
                      "Examination_ArmCircum", 
                      "Examination_WaistCircum", 
                      "Dietary_DietaryWeight", 
                      "Questionnaire_ConsiderWeight", 
                      "Questionnaire_SelfReportedWeight1YrAgo",
                      "Questionnaire_SelfReportedWeightAge25", 
                      "Questionnaire_SelfReportedWeight10YrAgo", 
                      "Questionnaire_DoctorOverweight", 
                      "Questionnaire_GeneralHealth",
                      "Examination_BMI"]

def load_nhanes_new(is_cycle_split=True,val_size=0.2):
    """
    Load nhanes data from 1999-2014.  
    Pre-processed data with imputation.
    """

    # Load X data
    X_fname  = "/projects/leelab2/wqiu/NHANES/data/"
    X_fname += "data_390_classification_imputed_missforest_feature_selection.csv"
    X = pd.read_csv(X_fname)

    # Five year mortality prediction
    year_num = 5
    
    X = X[X[str(year_num)+'_year_label']!=2]
    y = X[str(year_num)+'_year_label']
    
    # Drop extra labels
    X_release_cycle = X["Demographics_ReleaseCycle"]
    drop_list = ["mortstat", "permth_int", 
                 '1_year_label', '2_year_label', '3_year_label', 
                 '4_year_label', '5_year_label']
    X = X.drop(drop_list, axis=1)
    X = X.drop(extra_weight_feats, axis=1)

    # Split data set
    if is_cycle_split: # Split trval-test by release cycle
        X_trval = X[X_release_cycle <  6]
        X_test  = X[X_release_cycle >= 6]
        y_trval = y[X_release_cycle <  6]
        y_test  = y[X_release_cycle >= 6]
    else:              # Split trval-test by random train test split
        X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2, random_state=12312)

    X_train, X_valid, y_train, y_valid = train_test_split(X_trval, y_trval, test_size=val_size, random_state=4212312)
    
    return(X_train, X_valid, X_test, y_train, y_valid, y_test)



#################################
### Gene expression data sets ###
#################################
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def load_gene_expr(outcome_name):
    """
    Load gene expression data
    """
    DPATH = "data/"
    
    if outcome_name == "alzheimers":
        
        # Load data
        y_data = pd.read_csv(DPATH+"AD_DATA/ROSMAP_GE1_Diagnosis_Labels.tsv", sep='\t')
        X_data = pd.read_csv(DPATH+"AD_DATA/ROSMAP_GE1_Preprocessed_Expression_Matrix.tsv", sep='\t')

        # Drop subject ID
        X_data = X_data.drop(X_data.columns[0], axis=1) 
        y_data = y_data.drop(y_data.columns[0], axis=1)

        # Split data
        X_trval, X_test,  y_trval, y_test  = train_test_split(X_data, y_data, test_size=0.2, random_state=23)
        X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.3, random_state=12)

    elif "breastcancer" in outcome_name:
        
        # Load data
        y_data = pd.read_csv(DPATH+"CANCER_DATA/METABRIC_Labels.tsv", sep='\t')
        X_data = pd.read_csv(DPATH+"CANCER_DATA/METABRIC_Expression_NORMALIZED.tsv", sep='\t')

        # Drop subject ID
        X_data = X_data.drop(X_data.columns[0], axis=1) 

        if outcome_name == "breastcancer_er":
            # Use Estrogen Receptor Status phenotype as outcome
            y_data = y_data["ER Status"] == "Positive"
        elif outcome_name == "breastcancer_nhg":
            y_data = y_data["Neoplasm Histologic Grade"]        # Use tumor grade (from microscope) as outcome
            X_data = X_data.loc[np.invert(pd.isnull(y_data))]
            y_data = y_data[np.invert(pd.isnull(y_data))]
            y_data = y_data >= 3                                # Classify grade three tumors
        else:
            print("Failed to provide the specific breast cancer label")

        # Split data
        X_trval, X_test,  y_trval, y_test  = train_test_split(X_data, y_data, test_size=0.2, random_state=23)
        X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.3, random_state=12)

    return(X_train, X_valid, X_test, y_train, y_valid, y_test)

##############################
### FICO Lending club data ###
##############################
