import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Below is merely used for calculating metrics !!!
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

class LDA:
    def __init__(self):
        self.W = 0
        self.threshold = 0
        self.y1 = 0
        self.y2 = 0
        self.weight_y1_2 = 0
    
    # return weight vector W
    # X is features + label, y is only label
    def Fisher(self, X, y): 
        # deal with binary classification -- simplified case
        # target: m1, m2, S_B, S_W

        num_features = X.shape[1] - 1 # 64 in our case
        target_classes = np.unique(y)
        mean_vec = []
        S_W = np.zeros((num_features, num_features))
        
        for class_i in target_classes:
            print(class_i)
            # select only the related part
            X_i = X[X["class"] == class_i][X.columns[:-1]].to_numpy() # array[[]]
            mean_i = np.mean(X_i, axis=0) # m1 or m2
            mean_vec.append(mean_i) 
            S_W += np.outer(mean_i, mean_i)
            #print(S_W)

        # Binary classification -- simplified case
        m2_minus_m1 = np.subtract(mean_vec[0], mean_vec[1])
        S_B = np.outer(m2_minus_m1, m2_minus_m1)

        # S_B inverse * S_B
        S_W_inv = np.linalg.inv(S_W)
        S_W_inv_B = S_W_inv.dot(S_B)

        # eigenvalues & eigenvectors
        eig_vals, eig_vecs = np.linalg.eig(S_W_inv_B)  
        idx = eig_vals.argsort()[::-1] # reverse the order
        print(eig_vals)
        print(idx)
        eig_vals = eig_vals[idx] # Not needed
        print(eig_vals)
        # print(eig_vecs.shape)
        # print(eig_vecs ==  eig_vecs[:, idx])
        eig_vecs = eig_vecs[:, idx]
        # print(eig_vecs.shape)
        # print(eig_vecs)

        self.W = eig_vecs[0]
        self.threshold = 0.5 * np.dot(self.W, np.add(mean_vec[0], mean_vec[1]))
        self.y1 = np.dot(self.W, mean_vec[0])
        self.y2 = np.dot(self.W, mean_vec[1])

        #print("Weight vector is: ", self.W)
        print("Suspect threshold is: ", self.threshold)
        print("y1: ", self.y1)
        print("y2: ", self.y2)

        return eig_vecs   


    # X is only features
    def get_threshold(self, X, y_true, possible_threshold):
        #possible_threshold = [-0.002, -0.001, 0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0012, 0.0014, 0.0016, 0.0018, 0.002, 0.003, 0.004]
        print("training result")
        iter_f1 = []
        for threshold in possible_threshold:
            print("threhold", threshold)
            y_pred = []
            for i in range(len(X)):
                col = X[i]
                if ((threshold - np.dot(col, self.W)) * (threshold - self.y1) > 0):
                    y_pred.append(0)
                else:
                    y_pred.append(1)
            macro_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
            iter_f1.append(macro_f1)

            # macro
            macro = precision_recall_fscore_support(y_true, y_pred, average='macro')[:-1]
            print("For metrics = 'macro', precision, recall, f1 score, auc are: ", macro, roc_auc_score(y_true, y_pred), "respectivly")

        plt.plot(possible_threshold, iter_f1)
        plt.title("Threshold vs Macro F1")
        plt.xlabel('Threshold')
        plt.ylabel('Macro f1')
        plt.savefig("LDA", format="png")
        plt.show()

        k_idx= iter_f1.index(max(iter_f1))
        self.threshold = possible_threshold[k_idx]
        #self.weight_y1_2 = suspect_weight[k_idx]
        print("The chosen threshold is ", self.threshold)

    # X is only features
    def evaluation(self, X, y_true, possible_threshold):
        #num_features = X.shape[1] - 1 # 64 in our case
        # ----------------- change ------------------------# ----------------- change ------------------------
        target_classes = np.unique(y_true)
        mean_vec = []
        
        for class_i in target_classes:
            #print(class_i)
            # select only the related part
            X_i = X[X["class"] == class_i][X.columns[:-1]].to_numpy() # array[[]]
            mean_i = np.mean(X_i, axis=0) # m1 or m2
            mean_vec.append(mean_i) 

        y1 = np.dot(self.W, mean_vec[0])
        y2 = np.dot(self.W, mean_vec[1])
        print("y1", y1)

        X = X[X.columns[:-1]]
        X = X.to_numpy()

        # ----------------- change ------------------------# ----------------- change ------------------------
        print("testing result: ")
        #self.threshold = self.weight_y1_2 * y1 + ( 1- self.weight_y1_2) * y2
        for thres in possible_threshold:
            print(thres)
            y_pred = []
            
            for i in range(len(X)):
                col = X[i]
                if ((thres - np.dot(col, self.W)) * (thres - y1) > 0):
                    y_pred.append(0)
                else:
                    y_pred.append(1)
            
            # macro
            macro = precision_recall_fscore_support(y_true, y_pred, average='macro')[:-1]
            print("For metrics = 'macro', precision, recall, f1 score, auc are: ", macro, roc_auc_score(y_true, y_pred), "respectivly")

            # # other
            # micro = precision_recall_fscore_support(y_true, y_pred, average='micro')[:-1]
            # weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')[:-1]
            # print("For metrics = 'micro', precision, recall, f1 score are: ", micro, "respectivly")
            # print("For metrics = 'weighted', precision, recall, f1 score are: ", weighted, "respectivly")
            




def main():
    # training data 
    # smote: singular
    X_tr = pd.read_csv("./training_zscore_over.csv")
    y_tr = X_tr[X_tr.columns[-1]]

    # testing data
    X_ts = pd.read_csv("./testing_zscored.csv")
    y_ts = X_ts[X_ts.columns[-1]]
    #X_ts = X_ts[X_ts.columns[:-1]]
    
    # LDA starts
    possible_threshold = [-0.002, -0.001, 0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0012, 0.0014, 0.0016, 0.0018, 0.002, 0.003, 0.004]
    my_lda = LDA()
    my_lda.Fisher(X_tr, y_tr)
    my_lda.get_threshold(X_tr[X_tr.columns[:-1]].to_numpy(), y_tr, possible_threshold)
    my_lda.evaluation(X_ts, y_ts, possible_threshold)

main()