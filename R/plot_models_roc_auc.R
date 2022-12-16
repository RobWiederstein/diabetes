resamples <- readRDS("./data/model_resamples.rds")

lr_best <- select_best(resamples$lr, metric = "roc_auc")
knn_best <- select_best(resamples$knn, metric = "roc_auc")
rf_best <- select_best(resamples$rf, metric = "roc_auc")
xgb_best <- select_best(resamples$xgb, metric = "roc_auc")

models <- dplyr::bind_rows(
    #logistic regression results
    resamples$lr |>
    collect_predictions(parameters = lr_best) |>
    roc_curve(outcome, .pred_0) |>
    mutate(model = "Logistic Reg."),
    #nearest neighbor results
    resamples$knn |>
    collect_predictions(parameters = knn_best) |>
    roc_curve(outcome, .pred_0) |>
    mutate(model = "K-Nearest-Neighbor"),
    #random forest results
    resamples$rf |>
    collect_predictions(parameters = rf_best) |>
    roc_curve(outcome, .pred_0) |>
    mutate(model = "Random Forest"),
    #xgboost results
    resamples$xgb |>
    collect_predictions(parameters = xgb_best) |>
    roc_curve(outcome, .pred_0) |>
    mutate(model = "Xgboost")
)
saveRDS(models, file = "./data/model_performance.rds")
