import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import time
from sklearn.model_selection import RandomizedSearchCV

def classification(X_train, y_train):
    print("Running classification model ...")
    start = time.time()

    d_train = lgb.Dataset(X_train, label=y_train)
    params = {}
    params['learning_rate'] = 0.003
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = ''
    params['sub_feature'] = 0.5
    params['num_leaves'] = 10
    params['min_data'] = 50
    params['max_depth'] = 10

    clf = lgb.train(params, d_train, 100)

    end = time.time()
    print(end - start)
    return clf


def classification_RF(X_train, y_train):
    print("Running classification model ...")
    start = time.time()

    # random forest model creation
    rfc = RandomForestClassifier()
    clf = rfc.fit(X_train,y_train)
    print("clf.classes_", clf.classes_)

    end = time.time()
    print(end - start)
    return clf


def classification_RF_with_tunning(X_train, y_train,weights=None):
    print("Running classification model ...")

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 40, num = 10)] # 200, 2000, 100
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(1, 3, num = 3)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]


    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

    start = time.time()


    rfc = RandomForestClassifier()
    # random forest model creation
    rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 10,
                                 cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(X_train,y_train, sample_weight=weights)
    clf = rf_random.best_estimator_

    #clf = clf.fit(X_train,y_train)
    print("clf.classes_", clf.classes_)

    end = time.time()
    print(end - start)
    return clf
