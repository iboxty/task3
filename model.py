import numpy as np
from pandas import Series
import copy


class EMNaiveBayes(object):
    def __init__(self, eps):
        self.eps = eps
        # self.max_epoch = max_epoch

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
        # print self.X_onehot

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
        print self.X_onehot
        print self.A_onehot
        # print self.A
        # print self.X_number
        # print self.Pk

    def _update(self):
        self.Pk_old = copy.deepcopy(self.Pk)  # deep copy here
        self.A_onehot_old = copy.deepcopy(self.A_onehot)

        '''
        self.V = np.ones((self.num_exp, self.num_class))
        for j in range(self.num_fea):
            # print (self.A[j][:, self.X_number[:, j]]).T
            self.V *= (self.A[j][:, self.X_number[:, j]]).T
        # print self.V
        self.V *= self.Pk
        # print self.V
        self.V = self.V / np.sum(self.V, axis=1, keepdims=True)
        print self.V


        # update theta
        Ev_k = self.V.sum(axis=0).astype(float)
        self.pyk = Ev_k / float(self.num_exp)
        EvT = self.V.T
        col_idx = 0
        for j in range(self.num_fea):
            for l in range(self.num_val_per_fea[j]):
                self.A[j][:, l] = (EvT * self.X_onehot[:, col_idx + l]).sum(axis=1) / Ev_k
            col_idx += self.num_val_per_fea[j]
        print self.A
        '''

        self.V = np.ones((self.num_exp, self.num_class))
        for i in range(self.num_exp):
            for j in range(self.num_class):
                tmp = self.X_onehot[i] * self.A_onehot[j]
                mul = 1
                for num in tmp[tmp!=0]:
                    mul *= num
                self.V[i, j] = mul
        self.V *= self.Pk
        self.V = self.V / np.sum(self.V, axis=1, keepdims=True)
        # print self.V

        # update
        self.Pk = np.sum(self.V, axis=0) / self.num_exp
        # print self.Pk
        num_val_all = np.sum(self.num_val_per_fea)
        # print num_val_all
        for k in range(self.num_class):
            for j in range(num_val_all):
                self.A_onehot[k, j] = np.sum(self.X_onehot[:, j] * self.V[:, k]) / np.sum(self.V[:, k])
        # print self.A_onehot

    def _isconverge(self):
        diff = 0
        diff += np.sum((self.Pk_old - self.Pk) ** 2)
        diff += np.sum((self.A_onehot_old - self.A_onehot) ** 2)
        return diff < self.eps

    def fit(self, X, k, max_iter):
        self._train_init(X, k)
        for epoch in range(max_iter):
            self._update()
            if self._isconverge():
                break

    def _predict(self, X):
        # copy from other's code
        assert X.shape[1] == self.num_fea
        feature_list = [sorted(list(set(X[:, j]))) for j in range(self.num_fea)]
        num_val_per_fea = np.array([len(v) for v in feature_list])
        fea2num = [Series(range(num_val_per_fea[j]), index=feature_list[j]) for j in range(self.num_fea)]
        X_number = np.array([fea2num[j][X[:, j]] for j in range(self.num_fea)]).T
        X_onehot = np.zeros((X.shape[0], sum(num_val_per_fea)))
        k = 0
        for j in range(self.num_fea):
            for l in range(num_val_per_fea[j]):
                X_onehot[:, k + l] = (X_number[:, j] == l).astype(int)
            k += num_val_per_fea[j]
        pass




if __name__ == '__main__':
    # toy data as an example. For more, please go to main.py
    X = np.array([['A', 'B', 'A'], ['B', 'A', 'B'], ['B', 'A', 'C']])
    Y = np.array(['P', 'Q'])
    m = EMNaiveBayes(0.1)
    m._train_init(X, 2)
    m._update()
