from sklearn import svm
from sklearn import linear_model

# Shiyang add
from sklearn import lda
import numpy as np
from scipy import linalg
import scipy.io as sio

from pysofia import sofia_ml

import subprocess as sb
import os
import time


class linear_svm_lr(object):
    r"""
    Binary classifier that combines Linear Support Vector Machines and
    Logistic Regression.
    """
    def __init__(self, X, t):
        self.clf1 = svm.LinearSVC(class_weight='auto')
        self.clf1.fit(X, t)
        t1 = self.clf1.decision_function(X)
        self.clf2 = linear_model.LogisticRegression(class_weight='auto')
        self.clf2.fit(t1[..., None], t)

    def __call__(self, x):
        t1_pred = self.clf1.decision_function(x)
        return self.clf2.predict_proba(t1_pred[..., None])[:, 1]

class sofia_svm_pure(object):
    r"""
    Binary classifier that combines SVM (sofia-ml) and
    Logistic Regression.
    """
    def __init__(self, X, t):
        # if not contiguous in C, make it contiguous
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X, dtype=X.dtype)
        # if not contiguous in C, make it contiguous
        if not t.flags['C_CONTIGUOUS']:
            t = np.ascontiguousarray(t, dtype=t.dtype)

        # baseline: 0.00001, pegasos, balanced_stochastic, pegasos_eta, 10000
        self.lambda_ = 0.00001
        self.learner_ = sofia_ml.learner_type.pegasos
        self.loop_ = sofia_ml.loop_type.balanced_stochastic
        self.eta_ = sofia_ml.eta_type.pegasos_eta
        self.max_iter_ = 10000

        self.coef_ = sofia_ml.svm_train(X, t, None, self.lambda_, X.shape[0], X.shape[1],
                                  self.learner_, self.loop_, self.eta_, self.max_iter_)

    def __call__(self, x):
        # if not contiguous in C, make it contiguous
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x, dtype=x.dtype)
        prob = sofia_ml.svm_predict(x, self.coef_, sofia_ml.predict_type.logistic)

        return prob

    def update(self, X, t):
        # if not contiguous in C, make it contiguous
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X, dtype=X.dtype)
        # if not contiguous in C, make it contiguous
        if not t.flags['C_CONTIGUOUS']:
            t = np.ascontiguousarray(t, dtype=t.dtype)
        self.coef_ = sofia_ml.svm_update(X, t, self.coef_, self.lambda_, X.shape[0], X.shape[1],
                                         self.learner_, self.loop_, self.eta_, self.max_iter_)

class lda_lr(object):
    r"""
    Binary classifier that combines Linear Discriminant Analysis and
    Logistic Regression.
    """
    def __init__(self, X, t):
        self.clf1 = lda.LDA()
        self.clf1.fit(X, t)
        t1 = self.clf1.decision_function(X)

        # sio.savemat('C:\Csy\\t1.mat', {'t1': t1})

        self.clf2 = linear_model.LogisticRegression(class_weight='auto')
        self.clf2.fit(t1[..., None], t)

    def __call__(self, x):
        t1_pred = self.clf1.decision_function(x)
        return self.clf2.predict_proba(t1_pred[..., None])[:, 1]

class tk_lda_lr(object):
    r"""
    Binary classifier that combines Linear Discriminant Analysis (TK Kim's) and
    Logistic Regression.
    """
    def __init__(self, X, t):
        # sio.savemat('C:\Csy\X_t.mat', {'X': X, 't': t})
        self.clf1 = tk_LDA()
        self.clf1.fit(X, t)
        t1 = self.clf1.decision_function(X)
        self.clf2 = linear_model.LogisticRegression(class_weight='auto')
        self.clf2.fit(t1[..., None], t)

    def __call__(self, x):
        t1_pred = self.clf1.decision_function(x)
        return self.clf2.predict_proba(t1_pred[..., None])[:, 1]

    def update(self, X, t, prev_X, prev_t):
        self.clf1.increment(X, t)
        # t1 = self.clf1.decision_function(X)
        t1 = self.clf1.decision_function(np.vstack((X, prev_X)))
        self.clf2.fit(t1[..., None], np.vstack((t[:, None], prev_t[:, None]))[:, 0])

class tk_lda_pure(object):
    r"""
    Binary classifier that combines Linear Discriminant Analysis (TK Kim's).
    """
    def __init__(self, X, t):
        self.clf1 = tk_LDA()
        self.clf1.fit(X, t)

    def __call__(self, x):
        t1_pred = self.clf1.decision_function(x)
        prob = np.exp(t1_pred) / (1 + np.exp(t1_pred))
        return prob

    def update(self, X, t):
        self.clf1.increment(X, t)


class tk_LDA():
    """
    Linear Discriminant Analysis (LDA)
    classic textbook LDA algorithm
    Referred from T-K Kim's CVPR 2007 paper
    """
    def __init__(self):
        self.eigen_threshold = 1e-4
        self.d_eigen_threshold = 1e-4
        self.qr_threshold = 1e-4
        self.residue_threshold = 1e-4
        self.nonzero_threshold = 1e-4

    def fit(self, X, y):
        """
        Fit LDA model, refer to TK Kim's CVPR07 code
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array, shape = [n_samples]
            Target values (integers)
        :return:
        """
        # get eigenvectors and eigenvalues of the total and between-class matrix
        self.mean_, self.N_, self.t_eigvect_, self.t_eigval_ = self.get_st_model(X.T)
        _, _, self.b_eigvect_, self.b_eigval_, _, self.b_mean_pc_ = self.get_sb_model(X.T, y)
        # retrieve mean per class
        self.mean_pc_, self.Ns_, self.classes_ = self.get_mean(X.T, y)

        self.components_, self.eigval_ = self.get_components(self.t_eigvect_, self.t_eigval_, self.b_eigvect_, self.b_eigval_, self.N_)

        self.calc_classifier_param()

        # predict labels
        # prediction = self.predict_label(self.components_, X.T, self.mean_pc_, self.classes_)

        return self

    def increment(self, X, y):
        """
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array, shape = [n_samples]
            Target values (integers)
        :return:
        """
        # get eigenvectors and eigenvalues of the total and between-class matrix
        mean_2, N_2, t_eigvect_2, t_eigval_2 = self.get_st_model(X.T)
        _, _, b_eigvect_2, b_eigval_2, _, b_mean_pc_2 = self.get_sb_model(X.T, y)

        # retrieve mean per class
        mean_pc_2, Ns_2, classes_2 = self.get_mean(X.T, y)

        mean_new, N_new, t_eigvect_new, t_eigval_new = self.merge_st(
            self.mean_, self.N_, self.t_eigvect_, self.t_eigval_,
            mean_2, N_2, t_eigvect_2, t_eigval_2)

        _, _, b_eigvect_new, b_eigval_new, _, b_mean_pc_new = self.merge_sb(
            self.mean_, self.N_, self.b_eigvect_, self.b_eigval_, self.Ns_, self.b_mean_pc_, self.classes_,
            mean_2, N_2, b_eigvect_2, b_eigval_2, Ns_2, b_mean_pc_2, y)

        self.components_, _ = self.get_components(t_eigvect_new, t_eigval_new, b_eigvect_new, b_eigval_new, N_new)

        # update variables
        self.mean_ = mean_new
        self.N_ = N_new
        self.t_eigvect_ = t_eigvect_new
        self.t_eigval_ = t_eigval_new
        self.b_eigvect_ = b_eigvect_new
        self.b_eigval_ = b_eigval_new
        self.b_mean_pc_ = b_mean_pc_new
        self.mean_pc_ = (self.Ns_*self.mean_pc_ + Ns_2*mean_pc_2) / (self.Ns_+Ns_2)
        self.Ns_ += Ns_2
        if (self.classes_ != classes_2).any():
            raise ValueError("Class labels mismatched!")

        # update the classifier coeff as well
        self.calc_classifier_param()

    def merge_st(self, mean_1, N_1, t_eigvect_1, t_eigval_1, mean_2, N_2, t_eigvect_2, t_eigval_2):
        """
        Merge total scatter matrix
        :return:
        """
        # update global mean
        N_new = N_1 + N_2
        mean_new = (N_1*mean_1 + N_2*mean_2) / N_new

        G = t_eigvect_1.T.dot(t_eigvect_2)
        mean_diff = mean_1 - mean_2

        residue = t_eigvect_2 - t_eigvect_1.dot(G)
        residue_sum_row = np.sum(np.abs(residue), axis=0)
        sel_idx = residue_sum_row > self.residue_threshold
        pure_residue = residue[:, sel_idx]

        mean_residue = mean_diff - t_eigvect_1.dot(t_eigvect_1.T.dot(mean_diff))
        if len(mean_residue.shape) == 1:
            mean_residue = mean_residue[:, None]
        mean_residue_sum_row = np.sum(np.abs(mean_residue), axis=0)
        sel_idx = mean_residue_sum_row > self.residue_threshold
        mean_residue = mean_residue[:, sel_idx]

        orth_submatrix, upper_tri = np.linalg.qr(np.hstack((pure_residue, mean_residue)))
        # remove non-significant components
        upper_trisum = np.sum(np.abs(upper_tri), axis=1)
        sel_idx = upper_trisum > self.qr_threshold
        orth_submatrix = orth_submatrix[:, sel_idx]

        T = orth_submatrix.T.dot(t_eigvect_2)
        mG = t_eigvect_1.T.dot(mean_diff[:, None])
        mT = orth_submatrix.T.dot(mean_diff[:, None])

        reduce_dim = t_eigvect_1.shape[1] + orth_submatrix.shape[1]

        term1 = np.zeros((reduce_dim, reduce_dim))
        term1[:t_eigvect_1.shape[1], :t_eigvect_1.shape[1]] = t_eigval_1

        term2 = np.vstack( (np.hstack((G.dot(t_eigval_2).dot(G.T), G.dot(t_eigval_2).dot(T.T))),
                            np.hstack((T.dot(t_eigval_2).dot(G.T), T.dot(t_eigval_2).dot(T.T))) ) )

        term3 = np.vstack( (np.hstack((mG.dot(mG.T), mG.dot(mT.T))),
                            np.hstack((mT.dot(mG.T), mT.dot(mT.T))) ) ) * (N_1*N_2) / N_new

        composite = term1 + term2 + term3

        U, sigma, _ = np.linalg.svd(composite)

        sel_idx = sigma > self.eigen_threshold
        U = U[:, sel_idx]
        t_eigval_new = np.diag(sigma[sel_idx])
        t_eigvect_new = np.hstack((t_eigvect_1, orth_submatrix)).dot(U)

        return mean_new, N_new, t_eigvect_new, t_eigval_new

    def merge_sb(self, mean_1, N_1, b_eigvect_1, b_eigval_1, Ns_1, b_mean_pc_1, classes_1,
                 mean_2, N_2, b_eigvect_2, b_eigval_2, Ns_2, b_mean_pc_2, y_2):
        """
        Merge between-class scatter matrix
        :return:
        """
        # update global mean
        N_new = N_1 + N_2
        mean_new = (N_1*mean_1 + N_2*mean_2) / N_new
        dim = b_eigvect_1.shape[0]

        G = b_eigvect_1.T.dot(b_eigvect_2)
        mean_diff = mean_1 - mean_2
        residue = b_eigvect_2 - b_eigvect_1.dot(G)
        residue_sum_row = np.sum(np.abs(residue), axis=0)
        sel_idx = residue_sum_row > self.residue_threshold
        pure_residue = residue[:, sel_idx]

        mean_residue = mean_diff - b_eigvect_1.dot(b_eigvect_1.T.dot(mean_diff))
        if len(mean_residue.shape) == 1:
            mean_residue = mean_residue[:, None]
        mean_residue_sum_row = np.sum(np.abs(mean_residue), axis=0)
        sel_idx = mean_residue_sum_row > self.residue_threshold
        mean_residue = mean_residue[:, sel_idx]

        orth_submatrix, upper_tri = np.linalg.qr(np.hstack((pure_residue, mean_residue)))
        # remove non-significant components
        upper_trisum = np.sum(np.abs(upper_tri), axis=1)
        sel_idx = upper_trisum > self.qr_threshold
        orth_submatrix = orth_submatrix[:, sel_idx]
        # remove zero entry as well
        orth_sum = np.sum(np.abs(orth_submatrix), axis=0)
        nonzero_index = orth_sum > self.nonzero_threshold
        orth_submatrix = orth_submatrix[:, nonzero_index]

        # SVD
        T = orth_submatrix.T.dot(b_eigvect_2)
        mG = b_eigvect_1.T.dot(mean_diff[:, None])
        mT = orth_submatrix.T.dot(mean_diff[:, None])

        reduce_dim = b_eigvect_1.shape[1] + orth_submatrix.shape[1]

        term1 = np.zeros((reduce_dim, reduce_dim))
        term1[:b_eigvect_1.shape[1], :b_eigvect_1.shape[1]] = b_eigval_1

        term2 = np.vstack( (np.hstack((G.dot(b_eigval_2).dot(G.T), G.dot(b_eigval_2).dot(T.T))),
                            np.hstack((T.dot(b_eigval_2).dot(G.T), T.dot(b_eigval_2).dot(T.T))) ) )

        term4 = np.vstack( (np.hstack((mG.dot(mG.T), mG.dot(mT.T))),
                            np.hstack((mT.dot(mG.T), mT.dot(mT.T))) ) ) * (N_1*N_2) / N_new

        # for common classes
        term3 = np.zeros((reduce_dim, reduce_dim))
        labelset_2 = np.unique(y_2)
        labelset_com = np.intersect1d(classes_1, labelset_2)
        for i in range(labelset_com.size):
            idx1 = labelset_com[i] == classes_1
            idx2 = labelset_com[i] == labelset_2

            coeff = (-Ns_1[idx1]*Ns_2[idx2]) / (Ns_1[idx1]+Ns_2[idx2])
            classmean_diff = b_eigvect_1.dot(b_mean_pc_1[:, idx1]) + mean_1[:, None] - \
                             b_eigvect_2.dot(b_mean_pc_2[:, idx2]) - mean_2[:, None]

            cG = b_eigvect_1.T.dot(classmean_diff)
            cT = orth_submatrix.T.dot(classmean_diff)
            term3 += np.vstack( (np.hstack((cG.dot(cG.T), cG.dot(cT.T))),
                                 np.hstack((cT.dot(cG.T), cT.dot(cT.T))) ) ) * coeff

        composite = term1 + term2 + term3 + term4
        U, sigma, _ = np.linalg.svd(composite)

        sel_idx = sigma > self.eigen_threshold
        U = U[:, sel_idx]
        b_eigval_new = np.diag(sigma[sel_idx])
        b_eigvect_new = np.hstack((b_eigvect_1, orth_submatrix)).dot(U)

        # update other params
        labelset = np.unique(np.hstack((classes_1, y_2)))
        class_num = labelset.size
        Ns_new = np.zeros(class_num)
        b_mean_pc_new = np.zeros((b_eigvect_new.shape[1], class_num))

        for i in range(labelset_com.size):
            idx1 = labelset[i] == classes_1
            idx2 = labelset[i] == labelset_2
            submean_3 = np.zeros((dim, 1))

            if idx1.size > 0:
                Ns_new[i] += Ns_1[idx1]
                submean_3 += Ns_1[idx1]*(b_eigvect_1.dot(b_mean_pc_1[:, idx1]) + mean_1[:, None])

            if idx2.size > 0:
                Ns_new[i] += Ns_2[idx2]
                submean_3 += Ns_2[idx2]*(b_eigvect_2.dot(b_mean_pc_2[:, idx2]) + mean_2[:, None])

            submean_3 /= Ns_new[i]
            b_mean_pc_new[:, i] = b_eigvect_new.T.dot(submean_3 - mean_new[:, None])

        return mean_new, N_new, b_eigvect_new, b_eigval_new, Ns_new, b_mean_pc_new

    def calc_classifier_param(self):
        # set the params for decision function
        self.prior_ = self.Ns_/self.N_
        self.coef_  = self.components_.T.dot(self.mean_pc_ - self.mean_[:, None])
        self.intercept_ = -0.5*np.sum(self.coef_**2, axis=0) + np.log(self.prior_)

    def decision_function(self, X):
        """
        This function returns the decision function values related to each
        class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples, n_classes] or [n_samples,]
            Decision function values related to each class, per sample.
            In the two-class case, the shape is [n_samples,], giving the
            log likelihood ratio of the positive class.
        """

        dec_func = self._decision_function_Bayes(X.T)
        if len(self.classes_) == 2:
            return dec_func[:, 1] - dec_func[:, 0]
        return dec_func

    def _decision_function_Bayes(self, X):
        """
        Use Bayes rule to get the decision values for each class
        :return:
        """
        a = X - self.mean_[:, None]
        b = self.coef_.T.dot(self.components_.T).dot(a)
        decision_values = b.T + self.intercept_
        return decision_values

    def get_st_model(self, X):
        """
        total scatter matrix
        :param X:
        :return:
        """
        dim, N = X.shape
        eigenvector, eigenvalue, mean_vector, _ = self.tk_pca(X)
        return mean_vector, N, eigenvector, eigenvalue

    def get_sb_model(self, X, y):
        """
        between-class scatter matrix
        :param X:
        :param y:
        :return:
        """
        dim, N = X.shape

        # get unique class label
        label_set = np.unique(y)
        class_num = label_set.size

        mean_vector = X.mean(axis=1)
        sample_per_class = np.zeros((1, class_num))
        mean_per_class = np.zeros((dim, class_num))

        # mean per class
        for i in range(class_num):
            class_index = (y == label_set[i])
            sample_per_class[0, i] = np.sum(class_index)
            mean_per_class[:, i] = X[:, class_index].mean(axis=1)

        phi_mat = (mean_per_class - mean_vector[:, None]) * np.sqrt(sample_per_class)
        s_b = phi_mat.T.dot(phi_mat)

        u, s, _ = np.linalg.svd(s_b)
        # component selection
        sel_idx = s > self.eigen_threshold
        u = u[:, sel_idx]
        s = s[sel_idx]

        inv_sigma = np.diag(1/np.sqrt(s))
        eigenvector = phi_mat.dot(u.dot(inv_sigma))
        eigenvalue = np.diag(s)

        proj_mean_per_class = eigenvector.T.dot(mean_per_class - mean_vector[:, None])

        return mean_vector, N, eigenvector, eigenvalue, sample_per_class, proj_mean_per_class

    def get_mean(self, X, y):
        """
        retrieve mean per class
        :param X:
        :param y:
        :return:
        """
        dim, _ = X.shape

        # get unique class label
        label_set = np.unique(y)
        class_num = label_set.size

        sample_per_class = np.zeros(class_num)
        mean_per_class = np.zeros((dim, class_num))
        label_per_class = np.zeros(class_num)

        # mean per class
        for i in range(class_num):
            class_index = (y == label_set[i])
            sample_per_class[i] = np.sum(class_index)
            mean_per_class[:, i] = X[:, class_index].mean(axis=1)
            label_per_class[i] = label_set[i]

        return mean_per_class, sample_per_class, label_per_class

    def get_components(self, total_eigvect, total_eigval, between_eigvect, between_eigval, N):
        """
        get the discriminative components
        :return:
        """
        total_term_Z = total_eigvect.dot(np.linalg.inv(np.sqrt(total_eigval)))
        # you can control the components to keep, but I donnot here
        # total_term_Z = total_term_Z[:, :comp_num]

        spanset_tau, upper_tri = np.linalg.qr(total_term_Z.T.dot(between_eigvect))

        # remove non-significant components
        upper_trisum = np.sum(np.abs(upper_tri), axis=1)

        sel_idx = upper_trisum > self.qr_threshold
        spanset_tau = spanset_tau[:, sel_idx]

        half = spanset_tau.T.dot(total_term_Z.T.dot(between_eigvect))
        composite = half.dot(between_eigval).dot(half.T)

        u, s, _ = np.linalg.svd(composite)
        # component selection
        sel_idx = s > self.eigen_threshold
        u = u[:, sel_idx]
        s = s[sel_idx]

        components = total_term_Z.dot(spanset_tau.dot(u))

        return components, s

    def predict_label(self, transform_matrix, test, train, label_index):
        """
        Predict the labels based on euclidean distance
        :param transform_matrix:
        :param test:
        :param train:
        :param label_index:
        :return:
        """
        # project to LDA subspace
        proj_test = transform_matrix.T.dot(test)
        proj_mean = transform_matrix.T.dot(train)

        eucl_dist = np.tile(np.sum(proj_mean**2, axis=0), (proj_test.shape[1], 1)) + \
                    np.tile(np.sum(proj_test**2, axis=0)[:, None], (1, proj_mean.shape[1])) - \
                    2.0 * proj_test.T.dot(proj_mean)

        ind1 = np.argmin(eucl_dist, axis=1)
        prediction = label_index[ind1]

        return prediction

    def tk_pca(self, X):
        dim, N = X.shape
        # remove the mean
        mean_vector = X.mean(axis=1)
        data = X - mean_vector[:, None]

        # if feature dimension is larger
        if dim >= N:
            # get the covariance matrix
            covariance_mat_T = data.T.dot(data)/(N-1)
            u, s, _ = np.linalg.svd(covariance_mat_T)
            # component selection
            sel_idx = s > self.eigen_threshold
            u = u[:, sel_idx]
            s = s[sel_idx]
            # compose eigenvectors
            inv_sigma = np.diag(1/np.sqrt(s*(N-1)))
            eigenvector = data.dot(u.dot(inv_sigma))
            eigenvalue = np.diag(s)
        else:
            covariance_mat = data.dot(data.T)/(N-1)
            u, s, _ = np.linalg.svd(covariance_mat)
            # # component selection
            # sel_idx = s > self.eigen_threshold
            # u = u[:, sel_idx]
            # s = s[sel_idx]

            # select components based on percentage
            percentage = 0.98
            percents = np.cumsum(s)/np.sum(s)
            sel_idx = percents<=percentage
            u = u[:, sel_idx]
            s = s[sel_idx]

            eigenvector = u
            eigenvalue = np.diag(s)

        proj_data = eigenvector.T.dot(data)

        return eigenvector, eigenvalue, mean_vector, proj_data


# Shiyang add
def export_sofia_ml_data(filename, X, t, overwrite=False):
    # check label
    unique_label = np.unique(t)
    if not (unique_label==np.array([-1, 1])).all():
        raise ValueError("Labels has to be 1/-1 for each ")

    # save the txt file in sofia-ml svm data format
    if not overwrite:
        if os.path.exists(filename):
            raise ValueError(filename + " already exists!")

    with open(filename, 'w') as f:
        for i in range(len(t)):
            f.write(repr(int(t[i])) + " ")
            for j in range(X.shape[1]):
                if X[i, j] != 0:
                    f.write(repr(j) + ":" + repr(X[i, j]) + " ")
            f.write("\n")

# Shiyang add
def import_sofia_ml_results(filename, num):
    decision_values = np.zeros(num)

    i = 0
    with open(filename, 'r') as f:
        for line in f:
            decision_values[i] = np.float64(line.split('\t')[0])
            i += 1

    if i != num:
        raise ValueError("Num of instances is not equal to the SVM results!")

    return decision_values

# # dump code uses directly C code, extremely slow
# path = r"/data/RA/incremental-alignment/CLM/cache"
# # save the features
#
# dst_path = path + "/dim_%d_%d/" % (X.shape[0], X.shape[1]) + time.ctime()[:10].replace(" ", "_")
# if not os.path.exists(dst_path):
#     os.makedirs(dst_path)
#
# # id of files
# filename = dst_path + r"/" + time.ctime()[11:-4].replace(" ", "_") + repr(np.random.randint(0, 1000))
#
# train_path = filename + ".train"
# export_sofia_ml_data(train_path, X, t, overwrite=True)
#
# model_path = filename + ".model"
#
# orig_pwd = os.getcwd()
# code_path = r"/data/RA/incremental-alignment/incremental/sofia-ml-read-only"
# os.chdir(code_path)
#
# # call the c training code
# p = sb.Popen(["./sofia-ml", "--learner_type", "logreg-pegasos", "--loop_type", "balanced-stochastic",
#               "--lambda", "0.001", "--iterations", "200000", "--dimensionality", repr(X.shape[1]),
#               "--training_file", train_path, "--model_out", model_path],
#              stdout=sb.PIPE, stderr=sb.PIPE)
# p.communicate()
# p.wait()
#
# # # # just to test the train data
# # result_path = model_path.replace(".model", ".txt")
# # # call the c testing code
# # p = sb.Popen(["./sofia-ml", "--test_file", train_path, "--model_in", model_path,
# #               "--results_file", result_path, "--prediction_type", "logistic"],
# #              stdout=sb.PIPE, stderr=sb.PIPE)
# # p.communicate()
# # p.wait()
# # prob = import_sofia_ml_results(result_path, X.shape[0])
#
# os.chdir(orig_pwd)
#
# self.model_path = model_path
# self.code_path = code_path
#
# # delete temp training files
# if os.path.exists(train_path):
#     os.remove(train_path)
