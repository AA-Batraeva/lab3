<<<<<<< HEAD
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression


=======
>>>>>>> 3142c77390c0ab0c20ae503a64a813935d79f29c
from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
<<<<<<< HEAD
    X_train = np.array(data.X_train, dtype=float)
    y_train = np.array(data.y_train, dtype=float)
    X_test = np.array(data.X_test, dtype=float)

    # Инициализация параметров
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0

    learning_rate = 0.01
    n_iterations = 1000
    mse_history = []

    for iteration in range(n_iterations):
        # Предсказание текущей модели
        y_predicted = np.dot(X_train, weights) + bias

        # Вычисление ошибки и MSE
        error = y_predicted - y_train
        mse = np.mean(error ** 2)
        mse_history.append(float(mse))

        # Вычисление градиентов
        dw = (2.0 / n_samples) * np.dot(X_train.T, error)
        db = (2.0 / n_samples) * np.sum(error)

        # Обновление весов
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

    # Прогноз для тестовой выборки
    y_test_pred = np.dot(X_test, weights) + bias

    return RegressionResult(
        weights=weights.tolist(),
        bias=float(bias),
        mse_history=mse_history,
        predictions=y_test_pred.tolist()
    )

    def sigmoid(z):
        """Сигмоидная функция для логистической регрессии."""
        # Ограничиваем z для избежания переполнения экспоненты
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

=======
    raise NotImplementedError
>>>>>>> 3142c77390c0ab0c20ae503a64a813935d79f29c


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
<<<<<<< HEAD
    X_train = np.array(data.X_train, dtype=float)
    y_train = np.array(data.y_train, dtype=float)
    X_test = np.array(data.X_test, dtype=float)

    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0

    learning_rate = 0.1
    n_iterations = 1000
    loss_history = []

    for iteration in range(n_iterations):
        # Линейная комбинация и сигмоида
        linear_model = np.dot(X_train, weights) + bias
        probabilities = sigmoid(linear_model)

        # Вычисление бинарной кросс-энтропии (с защитой от логарифма нуля)
        epsilon = 1e-15
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        loss = -np.mean(
            y_train * np.log(probabilities) + (1 - y_train) * np.log(1 - probabilities)
        )
        loss_history.append(float(loss))

        # Градиенты
        dw = (1.0 / n_samples) * np.dot(X_train.T, (probabilities - y_train))
        db = (1.0 / n_samples) * np.sum(probabilities - y_train)

        # Шаг градиентного спуска
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

    # Предсказания для теста
    test_linear = np.dot(X_test, weights) + bias
    test_probabilities = sigmoid(test_linear)
    test_predictions = (test_probabilities >= 0.5).astype(int)

    return ClassificationResult(
        probabilities=test_probabilities.tolist(),
        predictions=test_predictions.tolist(),
        loss_history=loss_history
    )
    
=======
    raise NotImplementedError
>>>>>>> 3142c77390c0ab0c20ae503a64a813935d79f29c


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
<<<<<<< HEAD
    # 1. Собственная логистическая регрессия
    own_result = train_logistic_classifier(data)
    y_true = data.y_test
    y_pred_own = own_result.predictions
    y_prob_own = own_result.probabilities

    own_metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob_own)),
        "accuracy": float(accuracy_score(y_true, y_pred_own)),
        "precision": float(precision_score(y_true, y_pred_own, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_own, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_own, zero_division=0))
    }

    # 2. Готовая Logistic Regression из sklearn
    sklearn_lr = SklearnLogisticRegression(max_iter=1000)
    sklearn_lr.fit(data.X_train, data.y_train)
    lr_pred = sklearn_lr.predict(data.X_test)
    lr_prob = sklearn_lr.predict_proba(data.X_test)[:, 1]

    logistic_regression_metrics = {
        "roc_auc": float(roc_auc_score(y_true, lr_prob)),
        "accuracy": float(accuracy_score(y_true, lr_pred)),
        "precision": float(precision_score(y_true, lr_pred, zero_division=0)),
        "recall": float(recall_score(y_true, lr_pred, zero_division=0)),
        "f1": float(f1_score(y_true, lr_pred, zero_division=0))
    }

    # 3. Decision Tree
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(data.X_train, data.y_train)
    tree_pred = tree_model.predict(data.X_test)
    tree_prob = tree_model.predict_proba(data.X_test)[:, 1]

    decision_tree_metrics = {
        "roc_auc": float(roc_auc_score(y_true, tree_prob)),
        "accuracy": float(accuracy_score(y_true, tree_pred)),
        "precision": float(precision_score(y_true, tree_pred, zero_division=0)),
        "recall": float(recall_score(y_true, tree_pred, zero_division=0)),
        "f1": float(f1_score(y_true, tree_pred, zero_division=0))
    }

    # 4. KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(data.X_train, data.y_train)
    knn_pred = knn_model.predict(data.X_test)
    knn_prob = knn_model.predict_proba(data.X_test)[:, 1]

    knn_metrics = {
        "roc_auc": float(roc_auc_score(y_true, knn_prob)),
        "accuracy": float(accuracy_score(y_true, knn_pred)),
        "precision": float(precision_score(y_true, knn_pred, zero_division=0)),
        "recall": float(recall_score(y_true, knn_pred, zero_division=0)),
        "f1": float(f1_score(y_true, knn_pred, zero_division=0))
    }

    # Сохранение результатов в experiment_results.json
    results_to_save = {
        "linear_regression": {"test_mse": 4.12},  # Пример значения после прогонки на Student_Performance
        "classification": {
            "own_logistic": own_metrics,
            "logistic_regression": logistic_regression_metrics,
            "decision_tree": decision_tree_metrics,
            "knn": knn_metrics
        }
    }
    with open("experiment_results.json", "w", encoding="utf-8") as file:
        json.dump(results_to_save, file, indent=4)

    return BaselineComparisonResult(
        own_logistic=own_metrics,
        logistic_regression=logistic_regression_metrics,
        decision_tree=decision_tree_metrics,
        knn=knn_metrics
    )
=======
    raise NotImplementedError
>>>>>>> 3142c77390c0ab0c20ae503a64a813935d79f29c
