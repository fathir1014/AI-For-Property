from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from config.settings import MAX_ITER, ALPHA

def train_classifier(X_train, y_train):
    model = LogisticRegression(max_iter=MAX_ITER, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def train_regressors(X_train, y_train):
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=ALPHA),
        'lasso': Lasso(alpha=ALPHA)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models
