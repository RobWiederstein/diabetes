library(tidyverse)
library(tidymodels)
library(ggthemes)
library(glmnet)
# import ----
df  <- readr::read_csv(
    file = "./data-raw/diabetes.csv",
    col_names = c("pregn", "gluco",
                  "bp", "skint",
                  "insul", "bmi",
                  "dpf", "age", "outcome"
                  ),
    skip = 1
)
# set 0 to NA ----
df1 <-
    df |>
    mutate(across(c(gluco, bp, skint, insul, bmi), ~as.numeric(gsub(0, NA, .x))))
# impute ----
library(mice)
df1_imp <- mice(df1, m = 5, method = "pmm", seed = 123)
df2 <- complete(df1_imp, 3)
# plot distributions ----
library(ggplot2)
colors <- colorspace::qualitative_hcl(n = 8, palette = "Dark2")
df2 |>
    select(!outcome) |>
    mutate(across(pregn:age, ~scale(.x))) |>
    pivot_longer(pregn:age) |>
    group_by(name) |>
    mutate(median = median(value)) |>
    ungroup() |>
    mutate(name = factor(name)) |>
    mutate(name = forcats::fct_reorder(name, -median)) |>
    ggplot() +
    aes(name, value, group = name, fill = name) +
    geom_violin(alpha = .5, draw_quantiles = c(0.5)) +
    theme_tufte() +
    scale_fill_manual(values = colors) +
    labs(title = "Scaled Distribution of Diabetes Predictors",
         x = "",
         y = "") +
    theme(legend.position = "none")
# outcome to factor ----
df2$outcome <- factor(df2$outcome)

# begin models ----
##  split ----
library(tidymodels)
set.seed(502)
df_split <- initial_split(df2, prop = 0.80, strata = outcome)
df_train <- training(df_split)
df_test  <-  testing(df_split)
df_splits <- vfold_cv(df_train, strata = "outcome")
## model 1 - lr  .766 ----
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
        df_splits,
        lr_reg_grid,
        control = control_grid(save_pred = T),
        metrics = metric_set(accuracy, roc_auc)
    )
# select best params
lr_best <-
    lr_res |>
    select_best(metric = "roc_auc")
# create df w/ lr_auc
lr_auc <-
    lr_res %>%
    collect_predictions(parameters = lr_best) %>%
    roc_curve(outcome, .pred_0) %>%
    mutate(model = "Logistic Regression")
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
set.seed(345)
lr_last_fit <-
    lr_last_workflow %>%
    last_fit(df_split)
# confusion matrix
conf_mat(lr_last_fit[[5]][[1]], truth = outcome, estimate = .pred_class)
# Accuracy:
lr_res_acc <-
    lr_last_fit[[5]][[1]] |>
    rename(truth = outcome,
           predicted = .pred_class)
accuracy(lr_res_acc, truth, predicted) # .766
lr_last_fit[[3]][[1]]

## model 2 - knn .779 ----
# recipe
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
extract_parameter_set_dials(knn_mod)
# tune
set.seed(345)
knn_res <-
    knn_workflow %>%
    tune_grid(df_splits,
              grid = 25,
              control = control_grid(save_pred = TRUE),
              metrics = metric_set(accuracy, roc_auc))
# select best
# knn_best <-
#     knn_res %>%
#     select_best(metric = "roc_auc")
# knn_best
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
knn_final_wf <-
    knn_workflow |>
    update_model(knn_last_mod)

# Last Fit on train & applies to test!!!
knn_final_fit <-
    knn_final_wf |>
    last_fit(df_split)

# confusion matrix
knn_metrics <- knn_final_fit[[5]][[1]]
conf_mat(knn_metrics, truth = outcome, estimate = .pred_class)

knn_final_fit[[3]][[1]]
## model 3 - rf  .773 ----
# engine
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
    tune_grid(df_splits,
              grid = 25,
              control = control_grid(save_pred = TRUE),
              metrics = metric_set(accuracy, roc_auc))
#see results
rf_res %>%
    show_best(metric = "roc_auc")
autoplot(rf_res)
# pick results
rf_best <-
    rf_res %>%
    select_best(metric = "roc_auc")
# auc
rf_auc <-
    rf_res %>%
    collect_predictions(parameters = rf_best) |>
    roc_curve(outcome, .pred_0) %>%
    mutate(model = "Random Forest")
rf_auc |>
    ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) +
    geom_path(linewidth = 1.5, alpha = 0.8) +
    geom_abline(lty = 3) +
    coord_equal() +
    scale_color_viridis_d(option = "plasma", end = .6)
# last model
# the last model
rf_last_mod <-
    rand_forest(mtry = 5, min_n = 15, trees = 1000) %>%
    set_engine("ranger", num.threads = cores, importance = "impurity") %>%
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
# confustion matrix
conf_mat(rf_last_fit[[5]][[1]], truth = outcome, estimate = .pred_class)
rf_last_fit[[3]][[1]]
## model 4 - xgb .792 ----
# specify engine
xgb_spec <- boost_tree(
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
xgb_wf <-
    workflow() %>%
    add_formula(outcome ~ .) %>%
    add_model(xgb_spec)

# tune grid
doParallel::registerDoParallel()
set.seed(234)
xgb_res <- tune_grid(
    xgb_wf,
    resamples = df_splits,
    grid = xgb_grid,
    control = control_grid(save_pred = TRUE),
    metrics = metric_set(accuracy, roc_auc)
)
#visualize
xgb_res %>%
    collect_metrics() %>%
    filter(.metric == "roc_auc") %>%
    select(mean, mtry:sample_size) %>%
    pivot_longer(mtry:sample_size,
                 values_to = "value",
                 names_to = "parameter"
    ) %>%
    ggplot(aes(value, mean, color = parameter)) +
    geom_point(alpha = 0.8, show.legend = FALSE) +
    facet_wrap(~parameter, scales = "free_x") +
    labs(x = NULL, y = "AUC")
# best tuning
xgb_best_auc <- select_best(xgb_res, "roc_auc")
#finalize
xgb_final_fit <-
    finalize_workflow(
    xgb_wf,
    xgb_best_auc
)

# last fit
xgb_final_res <- last_fit(xgb_final_fit, df_split)

xgb_final_res %>%
    collect_predictions() %>%
    roc_curve(outcome, .pred_0) %>%
    ggplot(aes(x = 1 - specificity, y = sensitivity)) +
    geom_line(size = 1.5, color = "midnightblue") +
    geom_abline(
        lty = 2, alpha = 0.5,
        color = "gray50",
        size = 1.2
    )
# final results .7922
conf_mat(xgb_final_res[[5]][[1]], truth = outcome, estimate = .pred_class)
collect_metrics(xgb_final_res)
# end models ----
