def evaluate_models_comparatively(
    models,
    model_names,
    X_train,
    X_test,
    y_train,
    y_test,
    FEATURES,
    label_encoder,
    n_classes,
):
    print("Evaluando modelos comparativamente...\n")

    # Matriz de Confusión Normalizada comparativa
    plot_confusion_matrices(models, model_names, X_test, y_test, label_encoder)

    # Curva ROC comparativa
    plot_roc_curves(models, model_names, X_test, y_test, n_classes)

    # Curva de Aprendizaje comparativa
    plot_learning_curves(models, model_names, X_train, y_train)

    # Importancia de características comparativa
    plot_feature_importances(models, model_names, FEATURES)
