# import ----
diabetes <-readRDS("./data/diabetes.rds")
# split ----
set.seed(502)
df_split <- initial_split(diabetes, prop = 0.80, strata = outcome)
df_train <- training(df_split)
df_test  <-  testing(df_split)
df_folds <- vfold_cv(df_train, strata = "outcome")
# begin models -----
## lr ----
# recipe
lr_recipe <-
    recipe(formula = outcome ~ ., data = df_train) |>
    step_zv(all_predictors()) |>
    step_normalize()
# model
lr_mod <-
    logistic_reg(
        penalty = tune(),
        mixture = 1) %>%
    set_engine("glmnet")
# workflow
lr_workflow <-
    workflow() %>%
    add_recipe(lr_recipe) %>%
    add_model(lr_mod)
# create grid
lr_reg_grid <- tibble(penalty = 10^seq(-4, -1, length.out = 30))
# fit
lr_res <-
    lr_workflow |>
    tune_grid(
        df_folds,
        lr_reg_grid,
        control = control_grid(save_pred = T),
        metrics = metric_set(accuracy, roc_auc)
    )
# select best params
lr_best <- select_best(lr_res, metric = "roc_auc")
#last lr_fit
lr_last_mod <-
    logistic_reg(
        penalty = lr_best[[1, 1]],
        mixture = 1) %>%
    set_engine("glmnet")
# lr last workflow
lr_last_workflow <-
    lr_workflow %>%
    update_model(lr_last_mod)
# lr last fit
set.seed(345)
lr_last_fit <-
    lr_last_workflow %>%
    last_fit(df_split)
conf_mat(lr_last_fit[[5]][[1]], truth = outcome, estimate = .pred_class)
lr_last_fit[[3]][[1]]
## knn ----
knn_recipe <-
    recipe(formula = outcome ~ ., data = df_train) |>
    step_scale()
# model
knn_mod <-
    nearest_neighbor(
        mode = "classification",
        neighbors = tune(), #5
        weight_func = tune(), # "triangular"
        dist_power = tune() # 5
    ) %>%
    set_engine("kknn")
# workflow
knn_workflow <-
    workflow() %>%
    add_recipe(knn_recipe) %>%
    add_model(knn_mod)
# tune
set.seed(345)
knn_res <-
    knn_workflow %>%
    tune_grid(df_folds,
              grid = 25,
              control = control_grid(save_pred = TRUE),
              metrics = metric_set(accuracy, roc_auc))
knn_best <- select_best(knn_res, metric = "roc_auc")
# last model
knn_last_mod <-
    nearest_neighbor(
        mode = "classification",
        neighbors = 13, #5
        weight_func = "gaussian", # "triangular"
        dist_power = 1.08 # 5
    ) %>%
    set_engine("kknn")
# Finalize
knn_last_workflow <-
    knn_workflow |>
    update_model(knn_last_mod)

# Last Fit on train & applies to test!!!
knn_last_fit <-
    knn_last_workflow |>
    last_fit(df_split)
# confusion matrix
conf_mat(knn_last_fit[[5]][[1]], truth = outcome, estimate = .pred_class)
knn_last_fit[[3]][[1]]
## rf ----
cores <- parallel::detectCores()
rf_mod <-
    rand_forest(mtry = tune(), min_n = tune(), trees = 1000) |>
    set_engine("ranger", num.threads = cores) |>
    set_mode("classification")
# recipe
rf_recipe <- recipe(formula = outcome ~ ., data = df_train)
# workflow
rf_workflow <-
    workflow() %>%
    add_model(rf_mod) %>%
    add_recipe(rf_recipe)
# tune grid
set.seed(345)
rf_res <-
    rf_workflow %>%
    tune_grid(df_folds,
              grid = 25,
              control = control_grid(save_pred = TRUE),
              metrics = metric_set(accuracy, roc_auc))
# pick best
rf_best <- select_best(rf_res, metric = "roc_auc")
# the last model
rf_last_mod <-
    rand_forest(mtry = 4, min_n = 7, trees = 1000) %>%
    set_engine("ranger",
               num.threads = cores,
               importance = "impurity") %>%
    set_mode("classification")
# the last workflow
rf_last_workflow <-
    rf_workflow %>%
    update_model(rf_last_mod)
# the last fit
set.seed(345)
rf_last_fit <-
    rf_last_workflow %>%
    last_fit(df_split)
# confusion matrix
conf_mat(rf_last_fit[[5]][[1]], truth = outcome, estimate = .pred_class)
# metrics
rf_last_fit[[3]][[1]]
## xgboost ----
xgb_mod <- boost_tree(
    trees = 1000,
    tree_depth = tune(),
    min_n = tune(),
    loss_reduction = tune(), ## first three: model complexity
    sample_size = tune(),
    mtry = tune(), ## randomness
    learn_rate = tune(), ## step size
) %>%
    set_engine("xgboost") %>%
    set_mode("classification")

# set up grid
xgb_grid <- grid_latin_hypercube(
    tree_depth(),
    min_n(),
    loss_reduction(),
    sample_size = sample_prop(),
    finalize(mtry(), df_train),
    learn_rate(),
    size = 30
)

# create workflow
xgb_workflow <-
    workflow() %>%
    add_formula(outcome ~ .) %>%
    add_model(xgb_mod)

# tune grid
doParallel::registerDoParallel()
set.seed(234)
xgb_res <- tune_grid(
    xgb_workflow,
    resamples = df_folds,
    grid = xgb_grid,
    control = control_grid(save_pred = TRUE),
    metrics = metric_set(accuracy, roc_auc)
)
# best tuning
xgb_best <- select_best(xgb_res, "roc_auc")
xgb_best
#finalize
xgb_last_mod <-
    xgb_mod <- boost_tree(
        trees = 1000,
        tree_depth = 8,
        min_n = 4,
        loss_reduction = .00000118, ## first three: model complexity
        sample_size = .908,
        mtry = 4, ## randomness
        learn_rate = .00224, ## step size
    ) %>%
    set_engine("xgboost") %>%
    set_mode("classification")
# the last workflow
xgb_last_workflow <-
    xgb_workflow %>%
    update_model(xgb_last_mod)

# the last fit
xgb_last_fit <-
    xgb_last_workflow |>
    last_fit(df_split)
# conf
conf_mat(xgb_last_fit[[5]][[1]], truth = outcome, estimate = .pred_class)
# metrics
xgb_last_fit[[3]][[1]]
# end models ----
#
model_resamples <- list(
    knn = knn_res,
    lr = lr_res,
    rf = rf_res,
    xgb = xgb_res
)
test_set_results <- list(
    knn = knn_last_fit,
    lr = lr_last_fit,
    rf = rf_last_fit,
    xgb = xgb_last_fit

)
saveRDS(model_resamples, file = "./data/model_resamples.rds")
saveRDS(test_set_results, file = "./data/test_set_results.rds")

