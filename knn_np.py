import pandas as pd
import matplotlib.pyplot as plt

# Below is merely used for calculating metrics !!!
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Below is merely used for splitting data in cross validation part !!!
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=0)


# Below only converts pandas to numpy, to accelerate knn speed !!!
import numpy as np

def read():
    path_tr = "./training_zscore_smote.csv"
    path_ts = "./testing_zscored.csv"
    train = pd.read_csv(path_tr)
    test  = pd.read_csv(path_ts)
    train.drop(train.columns[0], axis=1, inplace=True)
    test.drop(test.columns[0], axis=1, inplace=True)
    trainX = train[train.columns[:-1]]
    trainY = train[train.columns[-1]]
    testX = test[test.columns[:-1]]
    testY = test[test.columns[-1]]

    return trainX.to_numpy(), trainY.to_numpy(), testX.to_numpy(), testY.to_numpy()


def distance(trainX, testX, tr_col, t_col, dis_metric):
    dis = 0
    dim = trainX.shape[1]


    if dis_metric == "Euclidean":
        a = trainX[tr_col]
        b = testX[t_col]
        return np.linalg.norm(a-b)
        
        for i in range(dim):
            #print(tr_col, t_col)
            dis += (trainX.iloc[tr_col, i] - testX.iloc[t_col, i])**2
        dis = dis**0.5
    elif dis_metric == "hamming":
        for i in range(dim):
            dis += abs(int(trainX.iloc[tr_col, i]) - int(testX.iloc[t_col, i]))
    
    return dis


# return pred results
def knn(k, trainX, testX, trainY, tr_idx, t_idx, dis_metric): # tr_idx is a list
    print("KNN for K = ",k)
    pred = []
    for t_col in t_idx:
    #for t_col in range(len(testX[0])):
        dis_to_all = [] # [dis1, .. , dis10]
        for tr_col in tr_idx:
        #for tr_col in range(len(trainX[0])):
            dis_to_all.append(distance(trainX, testX, tr_col, t_col, dis_metric))
        sorted_dis = [i[0] for i in sorted(enumerate(dis_to_all), key=lambda x:x[1])]
        
        # majority vote
        votes = {}
        for i in sorted_dis[:k]:
            # ----------------- change ------------------------# ----------------- change ------------------------
            vote_for = trainY[tr_idx, ][i]  # ----------------- change ---------------------------------------------------
            #print(vote_for)
            if vote_for in votes.keys():
                votes[vote_for] += 1
            else:
                votes[vote_for] = 1
        pred.append(max(votes, key = votes.get))
    print("When K = ", k, ", prediction is : ")
    #print(pred)
    return pred # record this! 

# ----------------- change ------------------------# ----------------- change ------------------------
def show_metrics(y_pred, true, true_idx): # true_idx is a list
    #print(pred)
    #print(true[0][true_idx])
    y_true = true[true_idx, ]

    # macro
    macro = precision_recall_fscore_support(y_true, y_pred, average='macro')[:-1]
    print("For metrics = 'macro', precision, recall, f1 score, auc are: ", macro, roc_auc_score(y_true, y_pred), "respectivly")

    # # other
    # micro = precision_recall_fscore_support(y_true, y_pred, average='micro')[:-1]
    # weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')[:-1]
    # print("For metrics = 'micro', precision, recall, f1 score are: ", micro, roc_auc_score(y_true, y_pred), "respectivly")
    # print("For metrics = 'weighted', precision, recall, f1 score are: ", weighted, roc_auc_score(y_true, y_pred), "respectivly")
    
    # # AUC
    # print("  For AUC, it's: ", roc_auc_score(y_true, y_pred))


# ----------------- change ------------------------# ----------------- change ------------------------
def train_k(trainX, trainY, file_name, dis_metric): # return k 
    suspect_k = int(len(trainX)**0.5)
    iter_k = []
    iter_f1= []
    
    print("Training starts, suspect_k is ", suspect_k)
    # Iterate k
    for i in range(min(2*suspect_k, 30)):
        if i%2 == 1:
            iter_k.append(i)
            print("Current iteration is k = ", i, "----------------")
            # Cross validation
            tem_ac = []   
            for train_index, test_index in kf.split(trainY):
                print("------ cross validation ------")
                # For debugging
                # print(train_index)
                # print(test_index)
                # ----------------- change ------------------------# ----------------- change ------------------------
                y_true = trainY[test_index, ] # pd column
                y_pred = knn(i, trainX, trainX, trainY, train_index, test_index, dis_metric) # list
                macro_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
                # ----------------- change ------------------------# ----------------- change ------------------------
                #acc = accuracy(knn(i, trainX, trainX, trainY, train_index, test_index, dis_metric), trainY, test_index)
                tem_ac.append(macro_f1)
            ac_f1 = sum(tem_ac)/len(tem_ac)
            iter_f1.append(ac_f1)
    
    # plots
    plt.plot(iter_k, iter_f1)
    plt.title("k vs macro f1")
    plt.xlabel('k')
    plt.ylabel('macro f1')
    plt.savefig(file_name, format="png")
    plt.show()
    #plt.close() # add

    # get k
    k_idx= iter_f1.index(max(iter_f1))
    k = iter_k[k_idx]

    print("The chosen k is ", k)
    knn(k, trainX, trainX, trainY, range(len(trainX)), range(len(trainX)), dis_metric)
    return k 


def predict(trainX, trainY, testX, testY, file_name, dis_metric):
    # k = train_k(trainX, trainY, file_name, dis_metric)
    # pred = knn(k, trainX, testX, trainY, range(len(trainX)), range(len(testX)), dis_metric)
    # show_metrics(pred, testY, range(len(testY)))
    for k in range(1, 33):
        if k%2 == 1:
            print("-------- k = ", k)
            print("---- 1. training: ")
            pred = knn(k, trainX, trainX, trainY, range(len(trainX)), range(len(trainX)), dis_metric)
            show_metrics(pred, trainY, range(len(trainY)))
            print("---- 2. testing: ")
            pred = knn(k, trainX, testX, trainY, range(len(trainX)), range(len(testX)), dis_metric)
            show_metrics(pred, testY, range(len(testY)))
            print("")


def main():
    trainX, trainY, testX, testY = read()
    predict(trainX, trainY, testX, testY, "fte.png", "Euclidean")

main()
