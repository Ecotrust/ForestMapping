import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets


# class OrdinalClassifier(BaseEstimator, ClassifierMixin):
#     """An ordinal classifier base class.
#     Adapted from https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c
#     """
#     def __init__(self, clf):
#         self.clf = clf
#         self.clfs = {}
    
#     def fit(self, X, y):
#         self.unique_class = np.sort(np.unique(y))
#         if self.unique_class.shape[0] > 2:
#             for i in range(self.unique_class.shape[0]-1):
#                 # for each k - 1 ordinal value we fit a binary classification problem
#                 binary_y = (y > self.unique_class[i]).astype(np.uint8)
#                 clf = clone(self.clf)
#                 clf.fit(X, binary_y)
#                 self.clfs[i] = clf
    
#     def predict_proba(self, X):
#         clfs_predict = {k:self.clfs[k].predict_proba(X) for k in self.clfs}
#         predicted = []
#         for i,y in enumerate(self.unique_class):
#             if i == 0:
#                 # V1 = 1 - Pr(y > V1)
#                 predicted.append(1 - clfs_predict[y][:,1])
#             elif y in clfs_predict:
#                 # Vi = Pr(y > Vi-1) - Pr(y > Vi)
#                  predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
#             else:
#                 # Vk = Pr(y > Vk-1)
#                 predicted.append(clfs_predict[y-1][:,1])
#         return np.vstack(predicted).T
    
#     def predict(self, X):
#         return np.argmax(self.predict_proba(X), axis=1)


class RandomForestOrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, *, criterion='gini', max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                 max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                 min_impurity_split=None, bootstrap=True, oob_score=False, 
                 n_jobs=None, random_state=None, verbose=0, 
                 warm_start=False, class_weight=None, ccp_alpha=0.0, 
                 max_samples=None):
        
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        
        self.clf_ = RandomForestClassifier(**self.get_params())
        self.clfs_ = {}
        self.classes_ = np.sort(np.unique(y))
        if self.classes_.shape[0] > 2:
            for i in range(self.classes_.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.classes_[i]).astype(np.uint8)
                clf = clone(self.clf_)
                clf.fit(X, binary_y)
                self.clfs_[i] = clf
        return self
    
    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        clfs_predict = {k:self.clfs_[k].predict_proba(X) for k in self.clfs_}
        predicted = []
        for i,y in enumerate(self.classes_):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T
    
    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        return np.argmax(self.predict_proba(X), axis=1)
    
class GradientBoostingOrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,*, loss='deviance', learning_rate=0.1, n_estimators=100, 
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2, 
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 
                 min_impurity_decrease=0.0, min_impurity_split=None, init=None, 
                 random_state=None, max_features=None, verbose=0, 
                 max_leaf_nodes=None, warm_start=False, presort='deprecated', 
                 validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, 
                 ccp_alpha=0.0):
        
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimator = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction = min_weight_fraction
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.init = init
        self.random_state = random_state
        self.max_features = max_features
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.presort = presort
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        
        self.clf_ = GradientBoostingClassifier(**self.get_params())
        self.clfs_ = {}
        self.classes_ = np.sort(np.unique(y))
        if self.classes_.shape[0] > 2:
            for i in range(self.classes_.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.classes_[i]).astype(np.uint8)
                clf = clone(self.clf_)
                clf.fit(X, binary_y)
                self.clfs_[i] = clf
        return self
    
    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        clfs_predict = {k:self.clfs_[k].predict_proba(X) for k in self.clfs_}
        predicted = []
        for i,y in enumerate(self.classes_):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T
    
    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        return np.argmax(self.predict_proba(X), axis=1)
        
class SVMOrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', 
                 coef0=0.0, shrinking=True, probability=True, tol=0.001, 
                 cache_size=200, class_weight='balanced', verbose=False, 
                 max_iter=-1, decision_function_shape='ovr', 
                 break_ties=False, random_state=None):
        
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        
        self.clf_ = SVC(**self.get_params())
        self.clfs_ = {}
        self.classes_ = np.sort(np.unique(y))
        if self.classes_.shape[0] > 2:
            for i in range(self.classes_.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.classes_[i]).astype(np.uint8)
                clf = clone(self.clf_)
                clf.fit(X, binary_y)
                self.clfs_[i] = clf
        return self
    
    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        clfs_predict = {k:self.clfs_[k].predict_proba(X) for k in self.clfs_}
        predicted = []
        for i,y in enumerate(self.classes_):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T
    
    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        return np.argmax(self.predict_proba(X), axis=1)
        
class KNeighborsOrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, *, weights='uniform', 
                 algorithm='auto', leaf_size=30, p=2, 
                 metric='minkowski', metric_params=None, n_jobs=None):
        
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        
        self.clf_ = KNeighborsClassifier(**self.get_params())
        self.clfs_ = {}
        self.classes_ = np.sort(np.unique(y))
        if self.classes_.shape[0] > 2:
            for i in range(self.classes_.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.classes_[i]).astype(np.uint8)
                clf = clone(self.clf_)
                clf.fit(X, binary_y)
                self.clfs_[i] = clf
        return self
    
    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        clfs_predict = {k:self.clfs_[k].predict_proba(X) for k in self.clfs_}
        predicted = []
        for i,y in enumerate(self.classes_):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T
    
    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        return np.argmax(self.predict_proba(X), axis=1)
        
        
class GaussianProcessOrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel=None, *, optimizer='fmin_l_bfgs_b', 
                 n_restarts_optimizer=0, max_iter_predict=100, 
                 warm_start=False, copy_X_train=True, 
                 random_state=None, multi_class='one_vs_rest', n_jobs=None):
        
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.multi_class= multi_class
        self.n_jobs = n_jobs
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        
        self.clf_ = GaussianProcessClassifier(**self.get_params())
        self.clfs_ = {}
        self.classes_ = np.sort(np.unique(y))
        if self.classes_.shape[0] > 2:
            for i in range(self.classes_.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.classes_[i]).astype(np.uint8)
                clf = clone(self.clf_)
                clf.fit(X, binary_y)
                self.clfs_[i] = clf
        return self
    
    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        clfs_predict = {k:self.clfs_[k].predict_proba(X) for k in self.clfs_}
        predicted = []
        for i,y in enumerate(self.classes_):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T
    
    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        return np.argmax(self.predict_proba(X), axis=1)
        
class LogisticRegressionOrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
                 intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', 
                 max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, 
                 l1_ratio=None):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        
        self.clf_ = LogisticRegression(**self.get_params())
        self.clfs_ = {}
        self.classes_ = np.sort(np.unique(y))
        if self.classes_.shape[0] > 2:
            for i in range(self.classes_.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.classes_[i]).astype(np.uint8)
                clf = clone(self.clf_)
                clf.fit(X, binary_y)
                self.clfs_[i] = clf
        return self
    
    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        clfs_predict = {k:self.clfs_[k].predict_proba(X) for k in self.clfs_}
        predicted = []
        for i,y in enumerate(self.classes_):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T
    
    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        return np.argmax(self.predict_proba(X), axis=1)