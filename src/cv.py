from sklearn.model_selection import StratifiedKFold

def StratifiedKFold_split(X, y, n_splits=4, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_index, val_index in skf.split(X, y):
        yield train_index, val_index
        