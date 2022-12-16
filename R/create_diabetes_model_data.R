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
df1_imp <- mice(df1, m = 5, method = "pmm", seed = 123)
df2 <- complete(df1_imp, 3)
# outcome to factor ----
df2$outcome <- factor(df2$outcome)
saveRDS(df2, "./data/diabetes.rds")
