from sklearn import svm
from sklearn import linear_model

# Shiyang add
from sklearn import lda
import numpy as np
from scipy import linalg
import scipy.io as sio


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
    Binary classifier that combines Linear Discriminant Analysis and
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


class tk_LDA():
    """
    Linear Discriminant Analysis (LDA)
    classic textbook LDA algorithm
    Referred from T-K Kim's CVPR 2007 paper
    """
    def __init__(self):
        self.eigen_threshold = 1e-4
        self.d_eigen_threshold = 1e-4

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return:
        """
        self.mean_, self.N_, self.t_eigvect_, self.t_eigval_ = self.get_st_model(X.T)
        _, _, self.b_eigvect_, self.b_eigval_, _, self.b_mean_pc_ = self.get_sb_model(X.T, y)
        # retrieve mean per class
        self.mean_pc_, self.Ns_, self.classes_ = self.get_mean(X.T, y)

        self.components_, self.eigval_ = self.get_components(self.t_eigvect_, self.t_eigval_, self.b_eigvect_, self.b_eigval_, self.N_)

        self.get_classifier()

        # predict labels
        # prediction = self.predict_label(self.components_, X.T, self.mean_pc_, self.classes_)

        return self

    def get_classifier(self):
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
        qr_threshold = 1e-4
        upper_trisum = np.sum(np.abs(upper_tri), axis=1)

        sel_idx = upper_trisum > qr_threshold
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
            # component selection
            sel_idx = s > self.eigen_threshold
            u = u[:, sel_idx]
            s = s[sel_idx]

            eigenvector = u
            eigenvalue = np.diag(s)

        proj_data = eigenvector.T.dot(data)

        return eigenvector, eigenvalue, mean_vector, proj_data