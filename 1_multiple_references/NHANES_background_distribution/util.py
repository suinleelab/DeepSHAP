from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from deepshap.data import preprocessed_data

def get_clusters(trainx, raw_trainx, is_plot=True, n_clusters=8):
    """
    Params
    ======
    
    trainx     : Train x data
    raw_trainx : Raw version of train x data
    is_plot    : Flag to denote whether we plot clusters
    n_clusters : Number of clusters to create
    
    Returns
    =======
    
    clusters     : List of train x clusters
    raw_clusters : List of raw train x clusters
    """
    
    # Get clusters (first two features are sex and age)
    kmeans_path = "models/kmeans_gender_age.p"
    if not os.path.exists(kmeans_path):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(trainx[:,0:2])
        pickle.dump(kmeans,open(kmeans_path, "wb"))
    else:
        kmeans = pickle.load(open(kmeans_path, "rb"))
    cluster_inds = kmeans.predict(trainx[:,0:2])
    clusters     = [trainx[cluster_inds==i] for i in range(n_clusters)]
    raw_clusters = [raw_trainx[cluster_inds==i] for i in range(n_clusters)]

    if is_plot:
        plt.rcParams["figure.figsize"] = (4,4)
        # Female age distributions
        for clusterx in raw_clusters:
            if clusterx["sex_isFemale"].mean() == 1:
                plt.hist(clusterx["age"])
        plt.ylabel("Count")
        plt.xlabel("Age")
        plt.title("Female age distributions")
        plt.tight_layout()
        plt.savefig("fig/female_clusters.pdf")
        plt.show()

        # Male age distributions
        for clusterx in raw_clusters:
            if clusterx["sex_isFemale"].mean() == 0:
                plt.hist(clusterx["age"])
        plt.ylabel("Count")
        plt.xlabel("Age")
        plt.title("Male age distributions")
        plt.tight_layout()
        plt.savefig("fig/male_clusters.pdf")
        plt.show()
    
    return(clusters, raw_clusters)

def get_raw_trainx():
    """
    Returns the raw train data
    """
    
    # Get the raw un-standardized data
    X, y = preprocessed_data()
    pids = np.unique(X.index.values)
    train_pids,test_pids = train_test_split(pids, random_state=0)
    strain_pids,valid_pids = train_test_split(train_pids, random_state=0)
    strain_inds = np.where([p in strain_pids for p in X.index.values])[0]
    raw_trainx = X.iloc[strain_inds,:]
    return(raw_trainx)

def cluster_summary_plots(clus_attr_lst, rand_attr_lst, clusters, raw_clusters):
    
    for i in range(len(clusters)):

        # Collect attributions and samples
        clus_attr = clus_attr_lst[i]
        rand_attr = rand_attr_lst[i]
        rtrainx2  = raw_clusters[i]
        trainx2   = clusters[i]

        # Determine gender/age range of cluster
        gender = "Male"
        if rtrainx2["sex_isFemale"].mean() == 1: 
            gender = "Female"

        age_min = rtrainx2["age"].min()
        age_max = rtrainx2["age"].max()

        # Print and plot results
        print("#"*20)
        print("### Gender: {}, Age: [{}, {}] ###".format(gender, age_min, age_max))
        print("#"*20)

        print("## Cluster Background ##")
        plt.clf()
        shap.summary_plot(clus_attr, features=trainx2, feature_names=feat_names, 
                          max_display=6, show=False)
        plt.savefig("fig/cluster_gender_{}_age_{}_{}.pdf".format(gender, age_min, age_max))
        plt.show()

        print("## Random Background ##")
        plt.clf()
        shap.summary_plot(rand_attr, features=trainx2, feature_names=feat_names, 
                          max_display=6, show=False)
        plt.savefig("fig/random_gender_{}_age_{}_{}.pdf".format(gender, age_min, age_max))
        plt.show()