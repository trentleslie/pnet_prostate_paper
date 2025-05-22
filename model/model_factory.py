import logging

from sklearn import svm, linear_model
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import Ridge, ElasticNet, Lasso, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from model import nn


# get a model object from a dictionary
# the params is in the format of {'type': 'model_type', 'params' {}}
# an example is params = {'type': 'svr', 'parmas': {'C': 0.025} }

def construct_model(model_params_dict):
    model_type = model_params_dict['type']
    p = model_params_dict['params']
    # logging.info ('model type: ', str(model_type))
    # logging.info('model paramters: {}'.format(p))

    if model_type == 'svr':
        model = svm.SVR(max_iter=5000, **p)

    if model_type == 'knn':
        model = KNeighborsClassifier(**p)

    if model_type == 'svc':
        model = svm.SVC(max_iter=5000, **p)

    if model_type == 'linear_svc':
        model = LinearSVC(max_iter=5000, **p)

    if model_type == 'multinomial':
        model = MultinomialNB(**p)

    if model_type == 'nearest_centroid':
        model = NearestCentroid(**p)

    if model_type == 'bernoulli':
        model = BernoulliNB(**p)

    if model_type == 'sgd':
        model = SGDClassifier(**p)

    if model_type == 'gaussian_process':
        model = GaussianProcessClassifier(**p)

    if model_type == 'decision_tree':
        model = DecisionTreeClassifier(**p)

    if model_type == 'random_forest':
        model = RandomForestClassifier(**p)

    if model_type == 'adaboost':
        model = AdaBoostClassifier(**p)

    if model_type == 'svr':
        model = svm.SVR(max_iter=5000, **p)
    # elif model_type == 'dt':
    #     # from sklearn.tree import DecisionTreeClassifier
    #     # model = DecisionTreeClassifier(**p)
    #     model = ModelWrapper(model)
    # elif model_type == 'rf':
    #     # from sklearn.ensemble import RandomForestClassifier
    #     model = RandomForestClassifier(**p)
    #     model = ModelWrapper(model)

    if model_type == 'ridge_classifier':
        model = RidgeClassifier(**p)

    elif model_type == 'ridge':
        model = Ridge(**p)


    elif model_type == 'elastic':
        model = ElasticNet(**p)
    elif model_type == 'lasso':
        model = Lasso(**p)
    elif model_type == 'randomforest':
        model = DecisionTreeRegressor(**p)

    elif model_type == 'extratrees':
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(**p)
        # print model

    elif model_type == 'randomizedLR':
        from sklearn.linear_model import RandomizedLogisticRegression
        model = RandomizedLogisticRegression(**p)

    elif model_type == 'AdaBoostDecisionTree':
        DT_params = params['DT_params']
        model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(**DT_params), **p)
    elif model_type == 'RandomForestRegressor':
        model = RandomForestRegressor(**p)
    elif model_type == 'ranksvm':
        model = RankSVMKernel()
    elif model_type == 'logistic':
        logging.info('model class {}'.format(model_type))
        model = linear_model.LogisticRegression()

    elif model_type == 'nn':
        # p is model_params_dict['params'] from YAML
        # It should contain 'build_fn_name', 'optimizer_type', 'learning_rate',
        # 'fitting_params', and a 'model_params' dict for the builder.

        p_copy = p.copy() # Work with a copy

        builder_name_str = p_copy.pop('build_fn_name', None)
        if not builder_name_str:
            raise ValueError("'build_fn_name' not found in nn model_params.params")

        actual_build_fn = None
        # Resolve builder_name_str to a function object
        # TODO: Implement a more robust dynamic lookup if many builders exist
        if builder_name_str == 'build_basic_nn': # Assuming maps to build_dense
            from model.builders import prostate_models
            actual_build_fn = prostate_models.build_dense
        elif builder_name_str == 'build_pnet':
            from model.builders import prostate_models
            actual_build_fn = prostate_models.build_pnet
        elif builder_name_str == 'build_pnet2':
            from model.builders import prostate_models
            actual_build_fn = prostate_models.build_pnet2
        # Add other known builders here
        else:
            # Fallback to dynamic import if name not explicitly mapped
            try:
                import importlib
                # This assumes builders are in 'model.builders.prostate_models'
                # Adjust if builders can be in other modules
                builders_module_path = 'model.builders.prostate_models'
                builders_module = importlib.import_module(builders_module_path)
                actual_build_fn = getattr(builders_module, builder_name_str)
            except (ImportError, AttributeError) as e:
                logging.error(f"Error dynamically loading builder '{builder_name_str}': {e}")
                raise ValueError(f"Unknown or unresolvable build_fn_name: {builder_name_str}")

        # Create optimizer object
        optimizer_type = p_copy.pop('optimizer_type', 'adam').lower()
        learning_rate = p_copy.pop('learning_rate', 0.001)
        
        # Ensure tensorflow is imported for optimizers
        import tensorflow as tf

        optimizer_obj = None
        if optimizer_type == 'adam':
            optimizer_obj = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer_obj = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_type == 'rmsprop':
            optimizer_obj = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        # Add other optimizers as needed
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

        # Ensure 'model_params' (for the builder) exists in p_copy and add optimizer to it
        # nn.Model.set_params expects builder args under sk_params['model_params']
        builder_specific_params = p_copy.get('model_params', {})
        builder_specific_params['optimizer'] = optimizer_obj # build_dense expects 'optimizer'
        p_copy['model_params'] = builder_specific_params
        
        # p_copy now contains remaining sk_params for nn.Model (e.g., fitting_params, model_params with optimizer)
        # actual_build_fn is the resolved function object
        model = nn.Model(actual_build_fn, **p_copy)

    return model


def get_model(params):
    if type(params['params']) == dict:
        model = construct_model(params)
    else:
        model = params['params']
    return model
