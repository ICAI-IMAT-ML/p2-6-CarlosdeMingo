import numpy as np

def cross_validation(model, X, y, nFolds):
    if nFolds == -1:
        # Leave-One-Out cross-validation
        nFolds = len(X)

    n_samples = len(X)
    accuracy_scores = []

    # Usamos indices secuenciales, sin shuffle, para mantener comportamiento original
    indices = np.arange(n_samples)
    fold_size = n_samples // nFolds

    for i in range(nFolds):
        # Último fold puede incluir más datos si no es divisible exacto
        start = i * fold_size
        end = (i + 1) * fold_size if i < nFolds - 1 else n_samples
        val_indices = indices[start:end]

        # Entrenamiento con los índices restantes
        train_indices = np.setdiff1d(indices, val_indices)

        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        accuracy_scores.append(score)

    return np.mean(accuracy_scores), np.std(accuracy_scores)
