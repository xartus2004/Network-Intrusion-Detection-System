import optuna
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC, LinearSVC
import joblib

def train_ensemble(X, Y):
    x_train, x_test, y_train, y_test = train_test_split_data(X, Y)
    
    # Load individual models
    knn = joblib.load('models/knn_model.pkl')
    lr = joblib.load('models/logistic_model.pkl')
    dt = joblib.load('models/decision_tree_model.pkl')
    rf = joblib.load('models/random_forest_model.pkl')
    xgb = joblib.load('models/xgboost_model.pkl')
    lgbm = joblib.load('models/lightgbm_model.pkl')
    cat = joblib.load('models/catboost_model.pkl')
    nb = joblib.load('models/naive_bayes_model.pkl')
    svm = joblib.load('models/svm_model.pkl')
    lsvc = joblib.load('models/linear_svc_model.pkl')

    # Create ensemble model
    ensemble = VotingClassifier(
        estimators=[
            ('knn', knn),
            ('lr', lr),
            ('dt', dt),
            ('rf', rf),
            ('xgb', xgb),
            ('lgbm', lgbm),
            ('cat', cat),
            ('nb', nb),
            ('svm', svm),
            ('lsvc', lsvc)
        ],
        voting='hard'
    )

    # Train ensemble model
    ensemble.fit(x_train, y_train)

    # Save ensemble model
    joblib.dump(ensemble, 'models/ensemble_model.pkl')

    return ensemble

def train_test_split_data(X, Y):
    return train_test_split(X, Y, train_size=0.70, random_state=42)

def train_knn(x_train, y_train, x_test, y_test):
    def objective(trial):
        n_neighbors = trial.suggest_int('KNN_n_neighbors', 2, 16)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(x_train, y_train)
        return model.score(x_test, y_test)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    best_model = KNeighborsClassifier(n_neighbors=study.best_trial.params['KNN_n_neighbors'])
    best_model.fit(x_train, y_train)
    
    joblib.dump(best_model, 'models/knn_model.pkl')
    return best_model.score(x_train, y_train), best_model.score(x_test, y_test)

def train_logistic_regression(x_train, y_train, x_test, y_test):
    def objective(trial):
        C = trial.suggest_loguniform('LR_C', 1e-5, 1e2)
        model = LogisticRegression(C=C, solver='liblinear')
        model.fit(x_train, y_train)
        return model.score(x_test, y_test)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    best_model = LogisticRegression(C=study.best_trial.params['LR_C'], solver='liblinear')
    best_model.fit(x_train, y_train)
    
    joblib.dump(best_model, 'models/logistic_model.pkl')
    return best_model.score(x_train, y_train), best_model.score(x_test, y_test)

def train_decision_tree(x_train, y_train, x_test, y_test):
    def objective(trial):
        max_depth = trial.suggest_int('DT_max_depth', 1, 20)
        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(x_train, y_train)
        return model.score(x_test, y_test)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    best_model = DecisionTreeClassifier(max_depth=study.best_trial.params['DT_max_depth'])
    best_model.fit(x_train, y_train)
    
    joblib.dump(best_model, 'models/decision_tree_model.pkl')
    return best_model.score(x_train, y_train), best_model.score(x_test, y_test)

def train_random_forest(x_train, y_train, x_test, y_test):
    def objective(trial):
        n_estimators = trial.suggest_int('RF_n_estimators', 10, 100)
        max_depth = trial.suggest_int('RF_max_depth', 1, 20)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(x_train, y_train)
        return model.score(x_test, y_test)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    best_model = RandomForestClassifier(n_estimators=study.best_trial.params['RF_n_estimators'],
                                         max_depth=study.best_trial.params['RF_max_depth'])
    best_model.fit(x_train, y_train)

    joblib.dump(best_model, 'models/random_forest_model.pkl')
    return best_model.score(x_train, y_train), best_model.score(x_test, y_test)

def train_xgboost(x_train, y_train, x_test, y_test):
    def objective(trial):
        n_estimators = trial.suggest_int('XGB_n_estimators', 10, 100)
        max_depth = trial.suggest_int('XGB_max_depth', 1, 20)
        learning_rate = trial.suggest_float('XGB_learning_rate', 0.01, 0.3)
        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
        model.fit(x_train, y_train)
        return model.score(x_test, y_test)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    best_model = XGBClassifier(n_estimators=study.best_trial.params['XGB_n_estimators'],
                                max_depth=study.best_trial.params['XGB_max_depth'],
                                learning_rate=study.best_trial.params['XGB_learning_rate'])
    best_model.fit(x_train, y_train)

    joblib.dump(best_model, 'models/xgboost_model.pkl')
    return best_model.score(x_train, y_train), best_model.score(x_test, y_test)

def train_lightgbm(x_train, y_train, x_test, y_test):
    def objective(trial):
        n_estimators = trial.suggest_int('LGBM_n_estimators', 10, 100)
        max_depth = trial.suggest_int('LGBM_max_depth', 1, 20)
        learning_rate = trial.suggest_float('LGBM_learning_rate', 0.01, 0.3)
        model = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
        model.fit(x_train, y_train)
        return model.score(x_test, y_test)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    best_model = LGBMClassifier(n_estimators=study.best_trial.params['LGBM_n_estimators'],
                                 max_depth=study.best_trial.params['LGBM_max_depth'],
                                 learning_rate=study.best_trial.params['LGBM_learning_rate'])
    best_model.fit(x_train, y_train)

    joblib.dump(best_model, 'models/lightgbm_model.pkl')
    return best_model.score(x_train, y_train), best_model.score(x_test, y_test)

def train_catboost(x_train, y_train, x_test, y_test):
    model = CatBoostClassifier(silent=True)
    model.fit(x_train, y_train)
    joblib.dump(model, 'models/catboost_model.pkl')
    return model.score(x_train, y_train), model.score(x_test, y_test)

def train_naive_bayes(x_train, y_train, x_test, y_test):
    model = BernoulliNB()
    model.fit(x_train, y_train)
    joblib.dump(model, 'models/naive_bayes_model.pkl')
    return model.score(x_train, y_train), model.score(x_test, y_test)

def train_svm(x_train, y_train, x_test, y_test):
    def objective(trial):
        C = trial.suggest_loguniform('SVC_C', 1e-5, 1e2)
        kernel = trial.suggest_categorical('SVC_kernel', ['linear', 'rbf', 'poly'])
        model = SVC(C=C, kernel=kernel)
        model.fit(x_train, y_train)
        return model.score(x_test, y_test)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    best_model = SVC(C=study.best_trial.params['SVC_C'], kernel=study.best_trial.params['SVC_kernel'])
    best_model.fit(x_train, y_train)

    joblib.dump(best_model, 'models/svm_model.pkl')
    return best_model.score(x_train, y_train), best_model.score(x_test, y_test)

def train_linear_svc(x_train, y_train, x_test, y_test):
    def objective(trial):
        C = trial.suggest_loguniform('LinearSVC_C', 1e-5, 1e2)
        model = LinearSVC(C=C, max_iter=10000)
        model.fit(x_train, y_train)
        return model.score(x_test, y_test)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    best_model = LinearSVC(C=study.best_trial.params['LinearSVC_C'], max_iter=10000)
    best_model.fit(x_train, y_train)

    joblib.dump(best_model, 'models/linear_svc_model.pkl')
    return best_model.score(x_train, y_train), best_model.score(x_test, y_test)

def train_all_models(X, Y):
    # Encode the target variable
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)

    # Assuming you're doing feature selection here
    # Add your feature selection code
    selected_features = ...  # Your feature selection logic
    
    # Save selected features
    joblib.dump(selected_features, 'models/selected_features.pkl')
    
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42)

    # Train models
    knn_train_scores = train_knn(x_train, y_train, x_test, y_test)
    lr_train_scores = train_logistic_regression(x_train, y_train, x_test, y_test)
    dt_train_scores = train_decision_tree(x_train, y_train, x_test, y_test)
    rf_train_scores = train_random_forest(x_train, y_train, x_test, y_test)
    xgb_train_scores = train_xgboost(x_train, y_train, x_test, y_test)
    lgbm_train_scores = train_lightgbm(x_train, y_train, x_test, y_test)
    cat_train_scores = train_catboost(x_train, y_train, x_test, y_test)
    nb_train_scores = train_naive_bayes(x_train, y_train, x_test, y_test)
    svm_train_scores = train_svm(x_train, y_train, x_test, y_test)
    lsvc_train_scores = train_linear_svc(x_train, y_train, x_test, y_test)

    results = {
        'KNN': knn_train_scores,
        'Logistic Regression': lr_train_scores,
        'Decision Tree': dt_train_scores,
        'Random Forest': rf_train_scores,
        'XGBoost': xgb_train_scores,
        'LightGBM': lgbm_train_scores,
        'CatBoost': cat_train_scores,
        'Naive Bayes': nb_train_scores,
        'SVM': svm_train_scores,
        'Linear SVC': lsvc_train_scores,
    }

    return results