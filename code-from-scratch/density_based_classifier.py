import scipy
from scipy import io
import matplotlib.pyplot as plt

# Below is merely used for splitting data in cross validation part !!!
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# ----------------- change ------------------------# ----------------- change ------------------------
# Be cautious about this !!! 
# Below is for generating random index of samples to calculate distances 
# => we get roughly get a range about proper radius, then use grid serach + cross validation to finalize r
import numpy as np # ----------------- change ------------------------
np.random.seed(42)

def read():
    # paths
    path_A_trainX = "./ATNT face/trainX.mat"
    path_A_trainY = "./ATNT face/trainY.mat"
    path_A_testX  = "./ATNT face/testX.mat"
    path_A_testY  = "./ATNT face/testY.mat"

    # data in the form of array
    A_trainX = scipy.io.loadmat(path_A_trainX)['trainX']
    A_trainY = scipy.io.loadmat(path_A_trainY)['trainY']
    A_testX  = scipy.io.loadmat(path_A_testX)['testX']
    A_testY  = scipy.io.loadmat(path_A_testY)['testY']

    return A_trainX, A_trainY, A_testX, A_testY


def read_bin():
    # paths
    path_A_trainX = "./Binalpha handwritten/trainX.mat"
    path_A_trainY = "./Binalpha handwritten/trainY.mat"
    path_A_testX  = "./Binalpha handwritten/testX.mat"
    path_A_testY  = "./Binalpha handwritten/testY.mat"

    # data in the form of array
    A_trainX = scipy.io.loadmat(path_A_trainX)['trainX']
    A_trainY = scipy.io.loadmat(path_A_trainY)['trainY']
    A_testX  = scipy.io.loadmat(path_A_testX)['testX']
    A_testY  = scipy.io.loadmat(path_A_testY)['testY']

    return A_trainX, A_trainY, A_testX, A_testY


def distance(trainX, testX, tr_col, t_col, dis_metric):
    dis = 0
    dim = len(trainX)
    # ----------------- change ------------------------
    #print("dim", dim)

    if dis_metric == "Euclidean":
        # ----------------- change ------------------------# ----------------- change ------------------------
        for i in range(dim):
            dis += (trainX[i][tr_col] - testX[i][t_col])**2
        dis = dis**0.5
    elif dis_metric == "hamming":
        for i in range(dim):
            dis += abs(int(trainX[i][tr_col]) - int(testX[i][t_col]))
    # if (dis >320):
    #     print(tr_col, t_col, dis)
    return dis


# return pred results
def density(k, trainX, testX, trainY, tr_idx, t_idx, dis_metric): # tr_idx is a list
    print("Density Based Classifier for Radius = ", k)
    pred = []
    num_empty = 0
    for t_col in t_idx:
    #for t_col in range(len(testX[0])):
        dis_to_all = [] # [dis1, .. , dis10]
        for tr_col in tr_idx:
        #for tr_col in range(len(trainX[0])):
            dis_to_all.append(distance(trainX, testX, tr_col, t_col, dis_metric))
        sorted_dis = [i[0] for i in sorted(enumerate(dis_to_all), key=lambda x:x[1])]
        # ----------------- change ------------------------
        #print(sorted_dis[:10]) # just indexes

        # majority vote
        votes = {}
        # ----------------- change ------------------------# ----------------- change ------------------------
        for i in sorted_dis:
            # within raius
            if i > k:
                if not bool(votes): # empty
                    #print("No point within the radius >...<")
                    # ----------------- change ------------------------# ----------------- change ------------------------
                    pred.append(0) # as a penalty
                    num_empty += 1
                    # vote_for = trainY[0][tr_idx][i]
                    # votes[vote_for] = 1
                break
            # back to usual
            vote_for = trainY[0][tr_idx][i]
            #print(vote_for)
            if vote_for in votes.keys():
                votes[vote_for] += 1
            else:
                votes[vote_for] = 1
        if bool(votes):
            pred.append(max(votes, key = votes.get))
    #print("When Raius = ", k, ", prediction is : ")
    #print(pred)
    return pred, num_empty # record this! 

# print & return accuracy
def accuracy(pred, true, true_idx): # true_idx is a list
    print(pred)
    print(true[0][true_idx])
    num_correct = 0
    for i in range(len(pred)):
        #print(pred[i], true[0][true_idx][i])  #   check more deeply ---------------------
        if pred[i] == true[0][true_idx][i]:
            num_correct +=1
    acc = num_correct/len(pred)
    print("Accuracy is: ", acc)
    return acc 

# ----------------- change ------------------------# ----------------- change ------------------------
def suspect_r(trainX, dis_metric):
    rand_list = np.random.choice(len(trainX.T), 20, replace=False)
    rand_dist = []
    for i in rand_list: # iter all points as anchor
        anchor_pt = i
        for j in rand_list: 
            if i == j:
                continue
            ship_pt = j
            dist = distance(trainX, trainX, anchor_pt, ship_pt, dis_metric)
            rand_dist.append(dist)
    avg_dis = sum(rand_dist)/len(rand_dist)
    #avg_dis = 320 # ----------------- change ------------------------
    print(avg_dis)
    suspect_r_list = []
    # ----------------- change ------------------------# ----------------- change ------------------------
    # Binaplha: [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 5]
    for i in [0.01, 0.1, 0.25, 0.5, 0.75, 1, 2]:
        suspect_r_list.append(i*avg_dis)
    
    return suspect_r_list


def train_r(trainX, trainY, file_name, dis_metric): # return r 
    # ----------------- change ------------------------# ----------------- change ------------------------
    suspect_k = suspect_r(trainX, dis_metric)
    iter_k = []
    iter_ac= []
    iter_empty = []
    iter_ac_cheat = []
    print("Training starts, suspect radius list is ", suspect_k)
    # Iterate radiua
    # ----------------- change ------------------------# ----------------- change ------------------------
    for i in suspect_k:
        iter_k.append(i)
        print("Current iteration is radius = ", i, "----------------")
        # Cross validation
        tem_ac = []
        tem_ac_ct = []
        tem_empty = []
        for train_index, test_index in kf.split(trainY[0]):
            print("------ cross validation ------")
            # ----------------- change ------------------------
            pred, num_emt = density(i, trainX, trainX, trainY, train_index, test_index, dis_metric)
            
            # consider invalid cases as wrong labelling
            acc = accuracy(pred, trainY, test_index)
            tem_ac.append(acc)
            # ignore invalid cases
            acc_ct = (acc * len(test_index))/(len(test_index) - num_emt)
            tem_ac_ct.append(acc_ct)
            tem_empty.append(num_emt/len(test_index))

        ac_k = sum(tem_ac)/len(tem_ac)
        iter_ac.append(ac_k)
        iter_ac_cheat.append(sum(tem_ac_ct)/len(tem_ac_ct))
        iter_empty.append(sum(tem_empty)/len(tem_empty))
    
    # plots 
    iter_0 = [0] * len(iter_k)

    plt.plot(iter_k, iter_ac, label="Accuracy with penalty to invalid cases")
    plt.plot(iter_k, iter_ac_cheat, label="Accuracy dimissing invalid cases")
    plt.plot(iter_k, iter_empty, label="Empty rate")
    plt.plot(iter_k, iter_0, label = "0")
    plt.legend()
    plt.title("Radius vs Accuracy & Empty rate")
    plt.xlabel('Radius')
    plt.ylabel('Accuracy & Empty rate')
    plt.savefig(file_name, format="png")
    plt.show()
    #plt.close() # add

    # get k
    k_idx= iter_ac.index(max(iter_ac))
    k = iter_k[k_idx]

    print("The chosen radius is ", k)
    return k 


def predict(trainX, trainY, testX, testY, file_name, dis_metric):
    k = train_r(trainX, trainY, file_name, dis_metric)
    pred, num_empty = density(k, trainX, testX, trainY, range(len(trainX[0])), range(len(testX[0])), dis_metric)
    accuracy(pred, testY, range(len(testY[0])))
    print("Empty rate is: ", num_empty/len(testX[0]))
    print("# empty is", num_empty)


def main():
    print("First with ATNT data: ")
    trainX, trainY, testX, testY = read()
    predict(trainX, trainY, testX, testY, "ATNT_2.png", "Euclidean")



    print("Then with Bin data: ")
    trainX, trainY, testX, testY = read_bin()
    predict(trainX, trainY, testX, testY, "Bin_2.png", "hamming")

    
main()
