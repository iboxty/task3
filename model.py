import numpy as np
from pandas import Series
import copy

class EMNaiveBayes():
    def __init__(self, eps):
        self.eps = eps

    def _train_init(self, input, num_class):
        '''
        :param input:   [[ 1.  0.  0.  1.  1.  0.  0.]
                        [ 0.  1.  1.  0.  0.  1.  0.]
                        [ 0.  1.  1.  0.  0.  0.  1.]]
        :param num_class: number of classes
        '''
        self.input = input
        self.num_exp = np.array(input).shape[0]
        self.num_fea = np.array(input).shape[1]
        self.num_class = num_class

        # copy from other's code
        feature_list = [sorted(list(set(input[:, j]))) for j in range(self.num_fea)]
        self.num_val_per_fea = np.array([len(v) for v in feature_list])
        fea2num = [Series(range(self.num_val_per_fea[j]), index=feature_list[j]) for j in range(self.num_fea)]
        self.X_number = np.array([fea2num[j][self.input[:, j]] for j in range(self.num_fea)]).T
        self.X_onehot = np.zeros((self.num_exp, sum(self.num_val_per_fea)))
        k = 0
        for j in range(self.num_fea):
            for l in range(self.num_val_per_fea[j]):
                self.X_onehot[:, k + l] = (self.X_number[:, j] == l).astype(int)
            k += self.num_val_per_fea[j]
        print self.X_onehot

        eps = 1e-2
        # initialize Pk
        tmp = np.random.rand(self.num_class) + eps
        self.Pk = tmp / np.sum(tmp)

        # initialize Aqjk
        self.A = list()
        for j in range(self.num_fea):
            fea_tmp = np.random.rand(self.num_class, self.num_val_per_fea[j]) + eps
            fea_tmp = fea_tmp / np.sum(fea_tmp, axis=1, keepdims=True)
            self.A.append(fea_tmp)

        self.A_onehot = np.concatenate(self.A, axis=1)
        # print input
        # print feature_list
        # print self.num_val_per_fea
        # print self.X_number
        # print self.X_onehot
        print self.A
        print self.X_number
        print self.Pk

    def _update(self):
        self.Pk_old = copy.deepcopy(self.Pk)  # deep copy here
        self.A_old = copy.deepcopy(self.A)

        self.V = np.ones((self.num_exp, self.num_class))
        '''
        for j in range(self.num_fea):
            # print (self.A[j][:, self.X_number[:, j]]).T
            self.V *= (self.A[j][:, self.X_number[:, j]]).T
        # print self.V
        self.V *= self.Pk
        # print self.V
        self.V = self.V / np.sum(self.V, axis=1, keepdims=True)
        print self.V
        '''



if __name__ == '__main__':
    # toy data as an example. For more, please go to main.py
    X = np.array([['A', 'B', 'A'], ['B', 'A', 'B'], ['B', 'A', 'C']])
    Y = np.array(['P', 'Q', 'I'])
    m = EMNaiveBayes(0.1)
    m._train_init(X, 3)
    m._update()
