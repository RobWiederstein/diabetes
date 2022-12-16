results <- readRDS("./data/test_set_results.rds")
results$knn |>
    collect_metrics()
metrics <- map_df(results, ~bind_rows(.x[[".metrics"]], .id = "model") )
metrics$model <- c("knn", "knn", "lr", "lr", "rf", "rf", "xgb", "xgb")
final <-
    metrics |>
    arrange(desc(.metric), desc(.estimate)) |>
    select(!.config)
saveRDS(final, "./table_final_results.rds")
