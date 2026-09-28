# ==============================================================================
# Feature selection, model development/evaluation, and SHAP analysis
# Public-sharing version: Chinese comments translated to English.
# Analysis logic is unchanged; only comments and the local absolute project path were edited.
# Set data_dir to the appropriate local project directory before running.
# ==============================================================================

# ==============================================================================
# CLHLS 2018
# Explainable machine-learning identification of cognitive impairment
# among older adults with anxiety symptoms
#
# Final machine-learning analysis
#
# 70 candidate predictors
#       ↓
# Stratified split:
# Development 80%
# Independent held-out test 20%
#       ↓
# Development only
#       ↓
# KNN imputation + scaling
#       ↓
# Elastic Net
#       ↓
# Random Forest-RFE
#       ↓
# Final predictor set
#       ↓
# Repeated 5-fold CV
#       ↓
# KNN imputation + scaling + SMOTE-Tomek
# within CV training folds only
#       ↓
# 9 base ML models
#       ↓
# OOF Stacking
#       ↓
# Independent test
#       ↓
# ROC / PR / Calibration / Lift
# AUROC / PR-AUC / Brier / classification metrics
#       ↓
# Kernel SHAP
#
# ==============================================================================



# ==============================================================================
# 0. Clear environment
# ==============================================================================

rm(list = ls())

gc()


options(
  stringsAsFactors = FALSE,
  scipen = 999
)


options(
  repos = c(
    CRAN = "https://cloud.r-project.org"
  )
)



# ==============================================================================
# 1. Parameters
# ==============================================================================

SEED <- 20260920

set.seed(SEED)


# Development 80%, test 20%
TRAIN_PROP <- 0.80


# Elastic Net
# Use lambda.min for the primary analysis
EN_RULE <- "lambda.min"

EN_ALPHA_GRID <- seq(
  0.1,
  1,
  by = 0.1
)

EN_FOLDS <- 10


# RFE
RFE_FOLDS <- 5

RFE_REPEATS <- 3


# Final machine-learning analysis
ML_FOLDS <- 5

ML_REPEATS <- 3


# Fixed classification threshold
CLASSIFICATION_THRESHOLD <- 0.50


# SHAP
RUN_SHAP <- TRUE

SHAP_BACKGROUND_N <- 100

SHAP_EXPLAIN_N <- 200



# ==============================================================================
# 2. R packages
# ==============================================================================

core_packages <- c(
  
  "caret",
  "caretEnsemble",
  
  "glmnet",
  
  "randomForest",
  "ranger",
  
  "rpart",
  
  "adabag",
  "gbm",
  
  "nnet",
  "kernlab",
  "naivebayes",
  
  "recipes",
  "themis",
  "RANN",
  
  "pROC",
  "PRROC",
  
  "ggplot2",
  
  "doParallel",
  "foreach"
)


shap_packages <- c(
  "kernelshap",
  "shapviz"
)



# ------------------------------------------------------------------------------
# Install missing core packages
# ------------------------------------------------------------------------------

missing_core <- core_packages[
  
  !vapply(
    core_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]


if (length(missing_core) > 0) {
  
  cat(
    "\n需要安装：\n"
  )
  
  print(
    missing_core
  )
  
  
  if (.Platform$OS.type == "windows") {
    
    install.packages(
      
      missing_core,
      
      type = "binary",
      
      dependencies = c(
        "Depends",
        "Imports"
      )
    )
    
  } else {
    
    install.packages(
      
      missing_core,
      
      dependencies = c(
        "Depends",
        "Imports"
      )
    )
  }
}



still_missing <- core_packages[
  
  !vapply(
    core_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]


if (length(still_missing) > 0) {
  
  stop(
    
    "以下包没有成功安装：\n",
    
    paste(
      still_missing,
      collapse = ", "
    )
  )
}



# ------------------------------------------------------------------------------
# SHAP packages
# ------------------------------------------------------------------------------

if (RUN_SHAP) {
  
  missing_shap <- shap_packages[
    
    !vapply(
      shap_packages,
      requireNamespace,
      logical(1),
      quietly = TRUE
    )
  ]
  
  
  if (length(missing_shap) > 0) {
    
    try(
      
      {
        
        if (.Platform$OS.type == "windows") {
          
          install.packages(
            
            missing_shap,
            
            type = "binary",
            
            dependencies = c(
              "Depends",
              "Imports"
            )
          )
          
        } else {
          
          install.packages(
            
            missing_shap,
            
            dependencies = c(
              "Depends",
              "Imports"
            )
          )
        }
      },
      
      silent = TRUE
    )
  }
  
  
  shap_available <- all(
    
    vapply(
      shap_packages,
      requireNamespace,
      logical(1),
      quietly = TRUE
    )
  )
  
  
  if (!shap_available) {
    
    warning(
      paste0(
        "kernelshap/shapviz未安装成功；",
        "前面的ML仍继续运行，仅跳过SHAP。"
      )
    )
    
    RUN_SHAP <- FALSE
  }
}



# ==============================================================================
# 3. Load packages
# ==============================================================================

library(caret)

library(caretEnsemble)

library(glmnet)

library(randomForest)

library(ranger)

library(rpart)

library(adabag)

library(gbm)

library(nnet)

library(kernlab)

library(naivebayes)

library(recipes)

library(themis)

library(RANN)

library(pROC)

library(PRROC)

library(ggplot2)

library(doParallel)

library(foreach)



# ==============================================================================
# 4. File paths
# ==============================================================================

data_dir <- "path/to/project"


data_path <- file.path(
  
  data_dir,
  
  "CLHLS_70变量_最终分析样本.csv"
)


result_dir <- file.path(
  
  data_dir,
  
  "Machine_learning_ElasticNet_RFE_SMOTETomek"
)


if (!dir.exists(result_dir)) {
  
  dir.create(
    result_dir,
    recursive = TRUE
  )
}


if (!file.exists(data_path)) {
  
  stop(
    "找不到数据：",
    data_path
  )
}



# ==============================================================================
# 5. Read data
# ==============================================================================

dat <- read.csv(
  
  data_path,
  
  fileEncoding = "UTF-8-BOM",
  
  stringsAsFactors = FALSE,
  
  check.names = FALSE
)


cat(
  "\n============================================\n"
)

cat(
  "数据读取完成\n"
)

cat(
  "============================================\n"
)


cat(
  "N =",
  nrow(dat),
  "\n"
)


cat(
  "字段数 =",
  ncol(dat),
  "\n"
)


if (nrow(dat) != 2081L) {
  
  warning(
    "当前样本量不是2081，当前N=",
    nrow(dat)
  )
}



# ==============================================================================
# 6. Fix the 70 candidate predictors
# ==============================================================================

predictor_names <- c(
  
  # 1-12
  "age_group",
  "gender",
  "nation",
  "good_health",
  "sleep_quality",
  "staple_food",
  "visual_impairment",
  "hearing_impairment",
  "physical_labor",
  "regular_physical_examination",
  "smoke",
  "drink_alcohol",
  
  # 13-25
  "fruit",
  "vegetable",
  "meat",
  "fish",
  "eggs",
  "bean_products",
  "salted_vegetables",
  "sugar",
  "garlic",
  "milk_products",
  "nut_products",
  "mushrooms_or_algae",
  "tea",
  
  # 26-29
  "indoor_ventilation_spring",
  "indoor_ventilation_summer",
  "indoor_ventilation_autumn",
  "indoor_ventilation_winter",
  
  # 30-50
  "hypertension",
  "diabetes",
  "heart_disease",
  "stroke_or_cvd",
  "chronic_lung_disease",
  "pulmonary_tuberculosis",
  "cataract",
  "glaucoma",
  "cancer",
  "prostate_tumor",
  "gastric_or_duodenal_ulcer",
  "parkinsons_disease",
  "bedsore",
  "arthritis",
  "epilepsy",
  "cholecystitis_cholelithiasis",
  "dyslipidemia",
  "rheumatism",
  "chronic_nephritis",
  "prostatic_hyperplasia",
  "hepatitis",
  
  # 51-52
  "fall",
  "toothache",
  
  # 53-56
  "energetically",
  "optimistic",
  "neatly",
  "working_condition",
  
  # 57-60
  "life_satisfaction",
  "read",
  "socialize",
  "marital_status",
  
  # 61-62
  "depression_risk",
  "disability_risk",
  
  # 63-70
  "residential_area",
  "hukou",
  "years_of_education",
  "residence_duration",
  "old_age_pension",
  "annual_household_income",
  "medical_insurance",
  "number_of_children"
)



if (length(predictor_names) != 70L) {
  
  stop(
    "候选变量不是70项。"
  )
}



missing_predictors <- setdiff(
  
  predictor_names,
  
  names(dat)
)


if (length(missing_predictors) > 0) {
  
  stop(
    
    "缺少变量：\n",
    
    paste(
      missing_predictors,
      collapse = ", "
    )
  )
}



if (!"cognitive_impairment" %in% names(dat)) {
  
  stop(
    "缺少 cognitive_impairment"
  )
}



# ==============================================================================
# 7. Save IDs
# ==============================================================================

if ("id" %in% names(dat)) {
  
  analysis_id <- as.character(
    dat$id
  )
  
} else {
  
  analysis_id <- as.character(
    seq_len(
      nrow(dat)
    )
  )
}



# ==============================================================================
# 8. Numeric conversion
# ==============================================================================

for (v in predictor_names) {
  
  dat[, v] <- suppressWarnings(
    
    as.numeric(
      dat[, v]
    )
  )
}


dat$cognitive_impairment <- suppressWarnings(
  
  as.numeric(
    dat$cognitive_impairment
  )
)



# ==============================================================================
# 9. Valid outcome
# ==============================================================================

valid_outcome <- dat$cognitive_impairment %in%
  c(
    0,
    1
  )


dat <- dat[
  valid_outcome,
  ,
  drop = FALSE
]


analysis_id <- analysis_id[
  valid_outcome
]



# ==============================================================================
# 10. Modeling dataset
# ==============================================================================

ml_data <- dat[
  c(
    predictor_names,
    "cognitive_impairment"
  )
]


# Yes must be the first factor level
ml_data$outcome <- factor(
  
  ifelse(
    
    ml_data$cognitive_impairment == 1,
    
    "Yes",
    
    "No"
  ),
  
  levels = c(
    "Yes",
    "No"
  )
)


ml_data$cognitive_impairment <- NULL



# ==============================================================================
# 11. Structural missingness:
# Prostatic disease in women = 0
# ==============================================================================

female_index <- (
  ml_data$gender == 2
)


ml_data$prostate_tumor[
  
  female_index &
    is.na(
      ml_data$prostate_tumor
    )
  
] <- 0


ml_data$prostatic_hyperplasia[
  
  female_index &
    is.na(
      ml_data$prostatic_hyperplasia
    )
  
] <- 0



# ==============================================================================
# 12. Outcome distribution
# ==============================================================================

cat(
  "\n结局分布：\n"
)


print(
  table(
    ml_data$outcome
  )
)


cat(
  "\n认知障碍比例 =",
  round(
    mean(
      ml_data$outcome == "Yes"
    ) * 100,
    2
  ),
  "%\n"
)



# ==============================================================================
# 13. Missingness rate
# ==============================================================================

missing_report <- data.frame(
  
  variable =
    predictor_names,
  
  missing_n =
    vapply(
      
      ml_data[
        predictor_names
      ],
      
      function(x)
        sum(
          is.na(x)
        ),
      
      numeric(1)
    ),
  
  missing_percent =
    round(
      
      100 *
        
        vapply(
          
          ml_data[
            predictor_names
          ],
          
          function(x)
            mean(
              is.na(x)
            ),
          
          numeric(1)
        ),
      
      2
    )
)


missing_report <- missing_report[
  order(
    -missing_report$missing_percent
  ),
  ,
  drop = FALSE
]


write.csv(
  
  missing_report,
  
  file.path(
    result_dir,
    "00_predictor_missingness.csv"
  ),
  
  row.names = FALSE
)



# ==============================================================================
# 14. Stratified random Development/Test split
# ==============================================================================

set.seed(
  SEED + 1
)


development_index <- caret::createDataPartition(
  
  ml_data$outcome,
  
  p = TRAIN_PROP,
  
  list = FALSE
)


development <- ml_data[
  development_index,
  ,
  drop = FALSE
]


test <- ml_data[
  -development_index,
  ,
  drop = FALSE
]


development_id <- analysis_id[
  development_index
]


test_id <- analysis_id[
  -development_index
]



cat(
  "\nDevelopment N =",
  nrow(development),
  "\n"
)


cat(
  "Test N =",
  nrow(test),
  "\n"
)


cat(
  "\nDevelopment：\n"
)


print(
  table(
    development$outcome
  )
)


cat(
  "\nTest：\n"
)


print(
  table(
    test$outcome
  )
)



# ==============================================================================
# 15. Save data split
# ==============================================================================

sample_split <- rbind(
  
  data.frame(
    
    id =
      development_id,
    
    dataset =
      "Development"
  ),
  
  data.frame(
    
    id =
      test_id,
    
    dataset =
      "Independent_test"
  )
)


write.csv(
  
  sample_split,
  
  file.path(
    result_dir,
    "01_sample_split_ID.csv"
  ),
  
  row.names = FALSE
)



# ==============================================================================
# 16. Development/Test X-Y
# ==============================================================================

dev_x_raw <- development[
  predictor_names
]


dev_y <- development$outcome


test_x_raw <- test[
  predictor_names
]


test_y <- test$outcome



# ==============================================================================
# 17. Check for completely missing variables
# ==============================================================================

all_missing <- names(
  dev_x_raw
)[
  
  vapply(
    
    dev_x_raw,
    
    function(x)
      all(
        is.na(x)
      ),
    
    logical(1)
  )
]


if (length(all_missing) > 0) {
  
  stop(
    
    "Development中完全缺失变量：",
    
    paste(
      all_missing,
      collapse = ", "
    )
  )
}



# ==============================================================================
# 18. Check for zero-variance variables
# ==============================================================================

zero_variance_info <- caret::nearZeroVar(
  
  dev_x_raw,
  
  saveMetrics = TRUE
)


zero_variance_variables <- rownames(
  zero_variance_info
)[
  zero_variance_info$zeroVar
]


if (length(zero_variance_variables) > 0) {
  
  stop(
    
    "发现零方差变量：",
    
    paste(
      zero_variance_variables,
      collapse = ", "
    )
  )
}



# ==============================================================================
# 19. Elastic Net / RFE preprocessing
#
# Development set only
# ==============================================================================

set.seed(
  SEED + 2
)


feature_preprocess <- caret::preProcess(
  
  dev_x_raw,
  
  method = c(
    "knnImpute",
    "center",
    "scale"
  ),
  
  k = 5
)


dev_x_feature <- predict(
  
  feature_preprocess,
  
  dev_x_raw
)



# ==============================================================================
# 20. Elastic Net
# ==============================================================================

x_en <- as.matrix(
  dev_x_feature
)


y_en <- ifelse(
  dev_y == "Yes",
  1,
  0
)



set.seed(
  SEED + 3
)


enet_foldid <- caret::createFolds(
  
  dev_y,
  
  k = EN_FOLDS,
  
  list = FALSE
)



enet_models <- vector(
  
  mode = "list",
  
  length =
    length(
      EN_ALPHA_GRID
    )
)


names(
  enet_models
) <- paste0(
  "alpha_",
  seq_along(
    EN_ALPHA_GRID
  )
)


enet_summary <- data.frame()



for (
  i in seq_along(
    EN_ALPHA_GRID
  )
) {
  
  current_alpha <-
    EN_ALPHA_GRID[i]
  
  
  cat(
    "Elastic Net alpha =",
    current_alpha,
    "\n"
  )
  
  
  set.seed(
    SEED + 100 + i
  )
  
  
  current_enet <- glmnet::cv.glmnet(
    
    x =
      x_en,
    
    y =
      y_en,
    
    family =
      "binomial",
    
    alpha =
      current_alpha,
    
    foldid =
      enet_foldid,
    
    type.measure =
      "auc",
    
    standardize =
      FALSE,
    
    nlambda =
      100
  )
  
  
  enet_models[i] <- list(
    current_enet
  )
  
  
  selected_lambda <- if (
    EN_RULE == "lambda.1se"
  ) {
    
    current_enet$lambda.1se
    
  } else {
    
    current_enet$lambda.min
  }
  
  
  lambda_position <- which.min(
    
    abs(
      current_enet$lambda -
        selected_lambda
    )
  )
  
  
  current_auc <- current_enet$cvm[
    lambda_position
  ]
  
  
  current_coef <- as.matrix(
    
    coef(
      current_enet,
      s =
        selected_lambda
    )
  )
  
  
  predictor_coef <- current_coef[
    
    rownames(
      current_coef
    ) !=
      "(Intercept)",
    
    1
  ]
  
  
  selected_n <- sum(
    predictor_coef != 0
  )
  
  
  enet_summary <- rbind(
    
    enet_summary,
    
    data.frame(
      
      alpha =
        current_alpha,
      
      lambda =
        selected_lambda,
      
      CV_AUC =
        current_auc,
      
      selected_n =
        selected_n
    )
  )
}



# ==============================================================================
# 21. Best Elastic Net model
# ==============================================================================

best_en_index <- which.max(
  enet_summary$CV_AUC
)


best_alpha <- enet_summary$alpha[
  best_en_index
]


best_enet_name <- paste0(
  "alpha_",
  best_en_index
)


best_enet <- getElement(
  
  enet_models,
  
  best_enet_name
)


best_lambda <- if (
  EN_RULE == "lambda.1se"
) {
  
  best_enet$lambda.1se
  
} else {
  
  best_enet$lambda.min
}



# ==============================================================================
# 22. Elastic Net features
# ==============================================================================

best_en_coef <- as.matrix(
  
  coef(
    best_enet,
    s =
      best_lambda
  )
)


enet_features <- rownames(
  best_en_coef
)[
  best_en_coef[, 1] != 0
]


enet_features <- setdiff(
  enet_features,
  "(Intercept)"
)



# lambda.min
coef_min <- as.matrix(
  
  coef(
    best_enet,
    s =
      best_enet$lambda.min
  )
)


features_min <- rownames(
  coef_min
)[
  coef_min[, 1] != 0
]


features_min <- setdiff(
  features_min,
  "(Intercept)"
)



# lambda.1se
coef_1se <- as.matrix(
  
  coef(
    best_enet,
    s =
      best_enet$lambda.1se
  )
)


features_1se <- rownames(
  coef_1se
)[
  coef_1se[, 1] != 0
]


features_1se <- setdiff(
  features_1se,
  "(Intercept)"
)



cat(
  "\n============================================\n"
)


cat(
  "Elastic Net结果\n"
)


cat(
  "============================================\n"
)


cat(
  "最佳alpha =",
  best_alpha,
  "\n"
)


cat(
  "lambda.min保留 =",
  length(
    features_min
  ),
  "\n"
)


cat(
  "lambda.1se保留 =",
  length(
    features_1se
  ),
  "\n"
)


cat(
  "主分析Elastic Net保留 =",
  length(
    enet_features
  ),
  "\n"
)


print(
  enet_features
)



write.csv(
  
  enet_summary,
  
  file.path(
    result_dir,
    "02_ElasticNet_summary.csv"
  ),
  
  row.names = FALSE
)


write.csv(
  
  data.frame(
    
    feature =
      enet_features,
    
    coefficient =
      best_en_coef[
        enet_features,
        1
      ]
  ),
  
  file.path(
    result_dir,
    "03_ElasticNet_selected_features.csv"
  ),
  
  row.names = FALSE
)



# ==============================================================================
# 23. RF-RFE
# ==============================================================================

n_en <- length(
  enet_features
)


if (n_en < 5) {
  
  stop(
    "Elastic Net特征过少，不执行RFE。"
  )
}



candidate_sizes <- unique(
  
  sort(
    
    c(
      5,
      10,
      15,
      20,
      25,
      30,
      35,
      40,
      n_en
    )
  )
)


candidate_sizes <- candidate_sizes[
  candidate_sizes <= n_en
]


candidate_sizes <- candidate_sizes[
  candidate_sizes >= 2
]



cat(
  "\nRFE候选特征数：\n"
)


print(
  candidate_sizes
)



rfe_functions <- caret::rfFuncs


rfe_functions$summary <-
  caret::twoClassSummary


rfe_functions$selectSize <-
  function(
    x,
    metric,
    maximize
  ) {
    
    caret::pickSizeTolerance(
      
      x,
      
      metric,
      
      tol = 1.5,
      
      maximize = maximize
    )
  }



rfe_control <- caret::rfeControl(
  
  functions =
    rfe_functions,
  
  method =
    "repeatedcv",
  
  number =
    RFE_FOLDS,
  
  repeats =
    RFE_REPEATS,
  
  verbose =
    TRUE,
  
  returnResamp =
    "final",
  
  allowParallel =
    FALSE
)



set.seed(
  SEED + 4
)


rfe_fit <- caret::rfe(
  
  x =
    dev_x_feature[
      enet_features
    ],
  
  y =
    dev_y,
  
  sizes =
    candidate_sizes,
  
  rfeControl =
    rfe_control,
  
  metric =
    "ROC",
  
  ntree =
    1000
)



rfe_features <- caret::predictors(
  rfe_fit
)



cat(
  "\n============================================\n"
)


cat(
  "RFE结果\n"
)


cat(
  "============================================\n"
)


cat(
  "Elastic Net =",
  n_en,
  "\n"
)


cat(
  "RFE最终 =",
  length(
    rfe_features
  ),
  "\n"
)


print(
  rfe_features
)



write.csv(
  
  rfe_fit$results,
  
  file.path(
    result_dir,
    "04_RFE_performance.csv"
  ),
  
  row.names = FALSE
)


write.csv(
  
  data.frame(
    feature =
      rfe_features
  ),
  
  file.path(
    result_dir,
    "05_RFE_final_features.csv"
  ),
  
  row.names = FALSE
)


saveRDS(
  
  rfe_fit,
  
  file.path(
    result_dir,
    "05_RFE_model.rds"
  )
)



# ==============================================================================
# 24. RFE curve
# ==============================================================================

p_rfe <- ggplot(
  
  rfe_fit$results,
  
  aes(
    x =
      Variables,
    
    y =
      ROC
  )
) +
  
  geom_line() +
  
  geom_point() +
  
  geom_vline(
    
    xintercept =
      length(
        rfe_features
      ),
    
    linetype = 2
  ) +
  
  labs(
    
    x =
      "Number of predictors",
    
    y =
      "Cross-validated AUROC",
    
    title =
      "Recursive Feature Elimination"
  ) +
  
  theme_classic()


ggsave(
  
  file.path(
    result_dir,
    "05_RFE_curve.png"
  ),
  
  p_rfe,
  
  width = 7,
  
  height = 5,
  
  dpi = 600
)



# ==============================================================================
# 25. Final machine-learning data
# ==============================================================================

dev_x <- development[
  rfe_features,
  drop = FALSE
]


dev_y <- development$outcome


test_x <- test[
  rfe_features,
  drop = FALSE
]


test_y <- test$outcome



# ==============================================================================
# 26. Custom SMOTE-Tomek
# ==============================================================================

smote_tomek <- list(
  
  name =
    "SMOTE_Tomek",
  
  
  func =
    function(
    x,
    y
    ) {
      
      temp <- as.data.frame(
        x
      )
      
      
      temp$.outcome <- y
      
      
      rec <- recipes::recipe(
        
        .outcome ~ .,
        
        data =
          temp
      )
      
      
      rec <- themis::step_smote(
        
        rec,
        
        .outcome,
        
        over_ratio =
          1,
        
        neighbors =
          5
      )
      
      
      rec <- themis::step_tomek(
        
        rec,
        
        .outcome
      )
      
      
      rec <- recipes::prep(
        
        rec,
        
        training =
          temp,
        
        retain =
          TRUE,
        
        verbose =
          FALSE
      )
      
      
      balanced <- recipes::juice(
        rec
      )
      
      
      balanced_x <- balanced[
        
        setdiff(
          names(balanced),
          ".outcome"
        )
      ]
      
      
      balanced_y <-
        balanced$.outcome
      
      
      list(
        
        x =
          balanced_x,
        
        y =
          balanced_y
      )
    },
  
  
  first =
    FALSE
)



# ==============================================================================
# 27. Final machine-learning cross-validation
# ==============================================================================

set.seed(
  SEED + 5
)


cv_index <- caret::createMultiFolds(
  
  dev_y,
  
  k =
    ML_FOLDS,
  
  times =
    ML_REPEATS
)



train_control <- caret::trainControl(
  
  method =
    "repeatedcv",
  
  number =
    ML_FOLDS,
  
  repeats =
    ML_REPEATS,
  
  index =
    cv_index,
  
  classProbs =
    TRUE,
  
  summaryFunction =
    caret::twoClassSummary,
  
  savePredictions =
    "final",
  
  sampling =
    smote_tomek,
  
  allowParallel =
    TRUE
)



# ==============================================================================
# 28. Nine base models
#
# Confirmed in the current environment:
#
# AdaBoost.M1 available
# naive_bayes available
# ==============================================================================

model_specs <- list(
  
  
  LR =
    caretEnsemble::caretModelSpec(
      
      method =
        "glm",
      
      family =
        binomial()
    ),
  
  
  KNN =
    caretEnsemble::caretModelSpec(
      
      method =
        "knn",
      
      tuneGrid =
        data.frame(
          
          k =
            seq(
              5,
              35,
              by = 2
            )
        )
    ),
  
  
  DecisionTree =
    caretEnsemble::caretModelSpec(
      
      method =
        "rpart",
      
      tuneLength =
        10
    ),
  
  
  AdaBoost =
    caretEnsemble::caretModelSpec(
      
      method =
        "AdaBoost.M1",
      
      tuneLength =
        5
    ),
  
  
  GradientBoosting =
    caretEnsemble::caretModelSpec(
      
      method =
        "gbm",
      
      tuneLength =
        5,
      
      verbose =
        FALSE
    ),
  
  
  RandomForest =
    caretEnsemble::caretModelSpec(
      
      method =
        "ranger",
      
      tuneLength =
        5,
      
      importance =
        "permutation",
      
      num.threads =
        1
    ),
  
  
  NeuralNetwork =
    caretEnsemble::caretModelSpec(
      
      method =
        "nnet",
      
      tuneLength =
        5,
      
      trace =
        FALSE,
      
      maxit =
        1000,
      
      MaxNWts =
        20000
    ),
  
  
  SVM =
    caretEnsemble::caretModelSpec(
      
      method =
        "svmRadial",
      
      tuneLength =
        7
    ),
  
  
  NaiveBayes =
    caretEnsemble::caretModelSpec(
      
      method =
        "naive_bayes",
      
      tuneLength =
        5
    )
)



# ==============================================================================
# 29. Parallel processing
# ==============================================================================

number_of_cores <- min(
  
  6,
  
  max(
    1,
    parallel::detectCores() -
      1
  )
)


cat(
  "\n使用CPU核心 =",
  number_of_cores,
  "\n"
)


cluster_object <- parallel::makePSOCKcluster(
  number_of_cores
)


parallel::clusterSetRNGStream(
  
  cluster_object,
  
  iseed =
    SEED + 10
)


doParallel::registerDoParallel(
  cluster_object
)



# ==============================================================================
# 30. Train nine models
# ==============================================================================

set.seed(
  SEED + 11
)


models <- caretEnsemble::caretList(
  
  x =
    dev_x,
  
  y =
    dev_y,
  
  trControl =
    train_control,
  
  tuneList =
    model_specs,
  
  metric =
    "ROC",
  
  preProcess = c(
    "knnImpute",
    "center",
    "scale"
  )
)



saveRDS(
  
  models,
  
  file.path(
    result_dir,
    "06_nine_base_models.rds"
  )
)



# ==============================================================================
# 31. Save best hyperparameters
# ==============================================================================

best_tuning_long <- data.frame()


for (
  model_name in names(
    models
  )
) {
  
  current_model <- getElement(
    
    models,
    
    model_name
  )
  
  
  current_tune <-
    current_model$bestTune
  
  
  if (
    is.null(
      current_tune
    ) ||
    ncol(
      current_tune
    ) == 0
  ) {
    
    current_long <- data.frame(
      
      Model =
        model_name,
      
      Parameter =
        NA_character_,
      
      Value =
        NA_character_
    )
    
  } else {
    
    current_long <- data.frame(
      
      Model =
        rep(
          model_name,
          ncol(
            current_tune
          )
        ),
      
      Parameter =
        names(
          current_tune
        ),
      
      Value =
        vapply(
          
          current_tune,
          
          function(z)
            paste(
              z,
              collapse = ";"
            ),
          
          character(1)
        )
    )
  }
  
  
  best_tuning_long <- rbind(
    
    best_tuning_long,
    
    current_long
  )
}



write.csv(
  
  best_tuning_long,
  
  file.path(
    result_dir,
    "07_best_tuning_parameters.csv"
  ),
  
  row.names = FALSE
)



# ==============================================================================
# 32. Base model CV results
# ==============================================================================

resamples_object <- caret::resamples(
  models
)


capture.output(
  
  summary(
    resamples_object
  ),
  
  file =
    file.path(
      result_dir,
      "08_base_models_CV_summary.txt"
    )
)


write.csv(
  
  resamples_object$values,
  
  file.path(
    result_dir,
    "08_base_models_CV_resamples.csv"
  ),
  
  row.names = FALSE
)



# ==============================================================================
# 33. Stacking
#
# IMPORTANT：
#
# outcome levels:
# 1 = Yes
# 2 = No
#
# excluded_class_id = 2L
#
# That is:
# Base models provide P(Yes) to the meta-learner
#
# This is more intuitive than excluding Yes by default.
# ==============================================================================

stack_control <- caret::trainControl(
  
  method =
    "cv",
  
  number =
    5,
  
  classProbs =
    TRUE,
  
  summaryFunction =
    caret::twoClassSummary,
  
  savePredictions =
    "final",
  
  allowParallel =
    TRUE
)



set.seed(
  SEED + 12
)


stack_model <- caretEnsemble::caretStack(
  
  models,
  
  method =
    "glm",
  
  family =
    binomial(),
  
  metric =
    "ROC",
  
  trControl =
    stack_control,
  
  excluded_class_id =
    2L
)



saveRDS(
  
  stack_model,
  
  file.path(
    result_dir,
    "09_stacking_model.rds"
  )
)


capture.output(
  
  summary(
    stack_model
  ),
  
  file =
    file.path(
      result_dir,
      "09_stacking_summary.txt"
    )
)



# ==============================================================================
# 34. Stop parallel processing
# ==============================================================================

parallel::stopCluster(
  cluster_object
)


foreach::registerDoSEQ()



# ==============================================================================
# 35. Base-model probability function
# ==============================================================================

predict_base_probability <- function(
    
  model,
  
  newdata
  
) {
  
  prob <- predict(
    
    model,
    
    newdata =
      newdata,
    
    type =
      "prob"
  )
  
  
  as.numeric(
    prob[, "Yes"]
  )
}



# ==============================================================================
# 36. Stacking probability function
#
# This is the most important correction in this revision.
#
# !!! Do NOT pass type="prob" to caretStack !!!
#
# predict.caretStack automatically returns probabilities for classification tasks.
#
# excluded_class_id = 2：
# Exclude No and return only the Yes probability.
# ==============================================================================

predict_stack_probability <- function(
    
  model,
  
  newdata
  
) {
  
  
  stack_prob <- stats::predict(
    
    model,
    
    newdata =
      newdata,
    
    excluded_class_id =
      2L
  )
  
  
  stack_prob <- as.data.frame(
    stack_prob
  )
  
  
  if (
    "Yes" %in%
    names(
      stack_prob
    )
  ) {
    
    return(
      
      as.numeric(
        stack_prob$Yes
      )
    )
  }
  
  
  stop(
    
    paste0(
      
      "Stacking预测没有找到Yes概率列。",
      
      "实际列名为：",
      
      paste(
        names(
          stack_prob
        ),
        collapse = ", "
      )
    )
  )
}



# ==============================================================================
# 37. Test-set prediction
# ==============================================================================

test_probabilities <- list()


for (
  model_name in names(
    models
  )
) {
  
  cat(
    "Independent test prediction：",
    model_name,
    "\n"
  )
  
  
  current_model <- getElement(
    
    models,
    
    model_name
  )
  
  
  current_probability <- predict_base_probability(
    
    current_model,
    
    test_x
  )
  
  
  test_probabilities[
    model_name
  ] <- list(
    current_probability
  )
}



cat(
  "Independent test prediction：Stacking\n"
)


stack_test_probability <- predict_stack_probability(
  
  stack_model,
  
  test_x
)


test_probabilities[
  "Stacking"
] <- list(
  stack_test_probability
)



# ==============================================================================
# 38. Development apparent predictions
#
# Used only as an overfitting check.
# ==============================================================================

development_probabilities <- list()


for (
  model_name in names(
    models
  )
) {
  
  current_model <- getElement(
    
    models,
    
    model_name
  )
  
  
  current_probability <- predict_base_probability(
    
    current_model,
    
    dev_x
  )
  
  
  development_probabilities[
    model_name
  ] <- list(
    current_probability
  )
}



development_probabilities[
  "Stacking"
] <- list(
  
  predict_stack_probability(
    
    stack_model,
    
    dev_x
  )
)



# ==============================================================================
# 39. Model-evaluation function
# ==============================================================================

evaluate_model <- function(
    
  truth,
  
  probability,
  
  threshold =
    0.50
  
) {
  
  truth_character <- as.character(
    truth
  )
  
  
  predicted_class <- factor(
    
    ifelse(
      
      probability >=
        threshold,
      
      "Yes",
      
      "No"
    ),
    
    levels = c(
      "Yes",
      "No"
    )
  )
  
  
  confusion <- caret::confusionMatrix(
    
    predicted_class,
    
    truth,
    
    positive =
      "Yes"
  )
  
  
  roc_truth <- factor(
    
    truth_character,
    
    levels = c(
      "No",
      "Yes"
    )
  )
  
  
  roc_object <- pROC::roc(
    
    response =
      roc_truth,
    
    predictor =
      probability,
    
    levels = c(
      "No",
      "Yes"
    ),
    
    direction =
      "<",
    
    quiet =
      TRUE
  )
  
  
  auc_value <- as.numeric(
    
    pROC::auc(
      roc_object
    )
  )
  
  
  auc_ci <- as.numeric(
    
    pROC::ci.auc(
      roc_object
    )
  )
  
  
  pr_object <- PRROC::pr.curve(
    
    scores.class0 =
      probability[
        truth_character ==
          "Yes"
      ],
    
    scores.class1 =
      probability[
        truth_character ==
          "No"
      ],
    
    curve =
      TRUE
  )
  
  
  pr_auc <-
    pr_object$auc.integral
  
  
  y01 <- ifelse(
    
    truth_character ==
      "Yes",
    
    1,
    
    0
  )
  
  
  brier <- mean(
    
    (
      y01 -
        probability
    )^2
  )
  
  
  # calibration
  probability_safe <- pmin(
    
    pmax(
      probability,
      0.000001
    ),
    
    0.999999
  )
  
  
  linear_predictor <- qlogis(
    probability_safe
  )
  
  
  calibration_intercept <- tryCatch(
    
    {
      
      fit_intercept <- glm(
        
        y01 ~
          offset(
            linear_predictor
          ),
        
        family =
          binomial()
      )
      
      
      as.numeric(
        coef(
          fit_intercept
        )[1]
      )
    },
    
    error =
      function(e)
        NA_real_
  )
  
  
  calibration_slope <- tryCatch(
    
    {
      
      fit_slope <- glm(
        
        y01 ~
          linear_predictor,
        
        family =
          binomial()
      )
      
      
      as.numeric(
        coef(
          fit_slope
        )[2]
      )
    },
    
    error =
      function(e)
        NA_real_
  )
  
  
  data.frame(
    
    AUC =
      auc_value,
    
    AUC_CI_low =
      auc_ci[1],
    
    AUC_CI_high =
      auc_ci[3],
    
    PR_AUC =
      pr_auc,
    
    Accuracy =
      as.numeric(
        confusion$overall[
          "Accuracy"
        ]
      ),
    
    Sensitivity =
      as.numeric(
        confusion$byClass[
          "Sensitivity"
        ]
      ),
    
    Specificity =
      as.numeric(
        confusion$byClass[
          "Specificity"
        ]
      ),
    
    Precision =
      as.numeric(
        confusion$byClass[
          "Pos Pred Value"
        ]
      ),
    
    Recall =
      as.numeric(
        confusion$byClass[
          "Sensitivity"
        ]
      ),
    
    F1 =
      as.numeric(
        confusion$byClass[
          "F1"
        ]
      ),
    
    Brier =
      brier,
    
    Calibration_intercept =
      calibration_intercept,
    
    Calibration_slope =
      calibration_slope
  )
}



# ==============================================================================
# 40. Test performance
# ==============================================================================

test_results <- data.frame()


for (
  model_name in names(
    test_probabilities
  )
) {
  
  current_probability <- getElement(
    
    test_probabilities,
    
    model_name
  )
  
  
  current_result <- evaluate_model(
    
    truth =
      test_y,
    
    probability =
      current_probability,
    
    threshold =
      CLASSIFICATION_THRESHOLD
  )
  
  
  current_result$Model <-
    model_name
  
  
  test_results <- rbind(
    
    test_results,
    
    current_result
  )
}



test_results <- test_results[
  
  c(
    
    "Model",
    
    setdiff(
      names(
        test_results
      ),
      "Model"
    )
  )
]


test_results <- test_results[
  
  order(
    -test_results$AUC
  ),
  
  ,
  
  drop = FALSE
]



cat(
  "\n============================================\n"
)


cat(
  "Independent Test Performance\n"
)


cat(
  "============================================\n"
)


print(
  test_results
)



write.csv(
  
  test_results,
  
  file.path(
    result_dir,
    "10_independent_test_performance.csv"
  ),
  
  row.names = FALSE
)



# ==============================================================================
# 41. Development apparent performance
# ==============================================================================

development_results <- data.frame()


for (
  model_name in names(
    development_probabilities
  )
) {
  
  current_probability <- getElement(
    
    development_probabilities,
    
    model_name
  )
  
  
  current_result <- evaluate_model(
    
    truth =
      dev_y,
    
    probability =
      current_probability,
    
    threshold =
      CLASSIFICATION_THRESHOLD
  )
  
  
  current_result$Model <-
    model_name
  
  
  development_results <- rbind(
    
    development_results,
    
    current_result
  )
}



development_results <- development_results[
  
  c(
    
    "Model",
    
    setdiff(
      names(
        development_results
      ),
      "Model"
    )
  )
]


write.csv(
  
  development_results,
  
  file.path(
    result_dir,
    "10a_development_apparent_performance.csv"
  ),
  
  row.names = FALSE
)



# ==============================================================================
# 42. Save test-set predicted probabilities
# ==============================================================================

test_prediction_output <- data.frame(
  
  id =
    test_id,
  
  truth =
    test_y
)


for (
  model_name in names(
    test_probabilities
  )
) {
  
  test_prediction_output[
    model_name
  ] <- getElement(
    
    test_probabilities,
    
    model_name
  )
}


write.csv(
  
  test_prediction_output,
  
  file.path(
    result_dir,
    "11_independent_test_probabilities.csv"
  ),
  
  row.names = FALSE
)



# ==============================================================================
# 43. ROC data
# ==============================================================================

roc_plot_data <- data.frame()


for (
  model_name in names(
    test_probabilities
  )
) {
  
  probability <- getElement(
    
    test_probabilities,
    
    model_name
  )
  
  
  roc_object <- pROC::roc(
    
    factor(
      
      as.character(
        test_y
      ),
      
      levels = c(
        "No",
        "Yes"
      )
    ),
    
    probability,
    
    quiet =
      TRUE
  )
  
  
  coordinates <- as.data.frame(
    
    pROC::coords(
      
      roc_object,
      
      x =
        "all",
      
      ret = c(
        "specificity",
        "sensitivity"
      ),
      
      transpose =
        FALSE
    )
  )
  
  
  coordinates$FPR <-
    1 -
    coordinates$specificity
  
  
  coordinates$TPR <-
    coordinates$sensitivity
  
  
  coordinates$Model <-
    model_name
  
  
  roc_plot_data <- rbind(
    
    roc_plot_data,
    
    coordinates[
      c(
        "FPR",
        "TPR",
        "Model"
      )
    ]
  )
}



# ==============================================================================
# 44. ROC
# ==============================================================================

p_roc <- ggplot(
  
  roc_plot_data,
  
  aes(
    
    x =
      FPR,
    
    y =
      TPR,
    
    group =
      Model,
    
    colour =
      Model
  )
) +
  
  geom_abline(
    
    intercept =
      0,
    
    slope =
      1,
    
    linetype =
      2
  ) +
  
  geom_line(
    linewidth =
      0.8
  ) +
  
  coord_equal() +
  
  labs(
    
    x =
      "1 - Specificity",
    
    y =
      "Sensitivity",
    
    title =
      "ROC Curves in the Independent Test Set"
  ) +
  
  theme_classic()



ggsave(
  
  file.path(
    result_dir,
    "12_test_ROC_curves.png"
  ),
  
  p_roc,
  
  width =
    8,
  
  height =
    7,
  
  dpi =
    600
)



# ==============================================================================
# 45. PR data
# ==============================================================================

pr_plot_data <- data.frame()


for (
  model_name in names(
    test_probabilities
  )
) {
  
  probability <- getElement(
    
    test_probabilities,
    
    model_name
  )
  
  
  truth_character <- as.character(
    test_y
  )
  
  
  pr_object <- PRROC::pr.curve(
    
    scores.class0 =
      probability[
        truth_character ==
          "Yes"
      ],
    
    scores.class1 =
      probability[
        truth_character ==
          "No"
      ],
    
    curve =
      TRUE
  )
  
  
  current_curve <- as.data.frame(
    pr_object$curve
  )
  
  
  if (ncol(current_curve) >= 2) {
    
    names(
      current_curve
    )[1:2] <- c(
      "Recall",
      "Precision"
    )
    
    
    current_curve$Model <-
      model_name
    
    
    pr_plot_data <- rbind(
      
      pr_plot_data,
      
      current_curve[
        c(
          "Recall",
          "Precision",
          "Model"
        )
      ]
    )
  }
}



# ==============================================================================
# 46. PR curve
# ==============================================================================

if (nrow(pr_plot_data) > 0) {
  
  p_pr <- ggplot(
    
    pr_plot_data,
    
    aes(
      
      x =
        Recall,
      
      y =
        Precision,
      
      group =
        Model,
      
      colour =
        Model
    )
  ) +
    
    geom_line(
      linewidth =
        0.8
    ) +
    
    coord_cartesian(
      
      xlim = c(
        0,
        1
      ),
      
      ylim = c(
        0,
        1
      )
    ) +
    
    labs(
      
      title =
        "Precision-Recall Curves in the Independent Test Set"
    ) +
    
    theme_classic()
  
  
  ggsave(
    
    file.path(
      result_dir,
      "13_test_PR_curves.png"
    ),
    
    p_pr,
    
    width =
      8,
    
    height =
      7,
    
    dpi =
      600
  )
}



# ==============================================================================
# 47. Calibration data
# ==============================================================================

calibration_data <- data.frame()


for (
  model_name in names(
    test_probabilities
  )
) {
  
  probability <- getElement(
    
    test_probabilities,
    
    model_name
  )
  
  
  y01 <- ifelse(
    test_y == "Yes",
    1,
    0
  )
  
  
  breaks <- unique(
    
    quantile(
      
      probability,
      
      probs =
        seq(
          0,
          1,
          length.out = 11
        ),
      
      na.rm =
        TRUE
    )
  )
  
  
  if (length(breaks) >= 3) {
    
    group <- cut(
      
      probability,
      
      breaks =
        breaks,
      
      include.lowest =
        TRUE,
      
      labels =
        FALSE
    )
    
    
    current_calibration <- aggregate(
      
      cbind(
        
        predicted =
          probability,
        
        observed =
          y01
      ),
      
      by =
        list(
          group =
            group
        ),
      
      FUN =
        mean
    )
    
    
    current_calibration$Model <-
      model_name
    
    
    calibration_data <- rbind(
      
      calibration_data,
      
      current_calibration
    )
  }
}



write.csv(
  
  calibration_data,
  
  file.path(
    result_dir,
    "14_test_calibration_data.csv"
  ),
  
  row.names = FALSE
)



# ==============================================================================
# 48. Calibration plot
# ==============================================================================

p_calibration <- ggplot(
  
  calibration_data,
  
  aes(
    
    x =
      predicted,
    
    y =
      observed,
    
    group =
      Model,
    
    colour =
      Model
  )
) +
  
  geom_abline(
    
    slope =
      1,
    
    intercept =
      0,
    
    linetype =
      2
  ) +
  
  geom_line() +
  
  geom_point() +
  
  coord_equal(
    
    xlim = c(
      0,
      1
    ),
    
    ylim = c(
      0,
      1
    )
  ) +
  
  labs(
    
    x =
      "Mean predicted probability",
    
    y =
      "Observed proportion",
    
    title =
      "Calibration Curves in the Independent Test Set"
  ) +
  
  theme_classic()



ggsave(
  
  file.path(
    result_dir,
    "14_test_calibration_curves.png"
  ),
  
  p_calibration,
  
  width =
    8,
  
  height =
    7,
  
  dpi =
    600
)



# ==============================================================================
# 49. Lift data
# ==============================================================================

lift_data <- data.frame()


for (
  model_name in names(
    test_probabilities
  )
) {
  
  probability <- getElement(
    
    test_probabilities,
    
    model_name
  )
  
  
  y01 <- ifelse(
    test_y == "Yes",
    1,
    0
  )
  
  
  ordering <- order(
    
    probability,
    
    decreasing =
      TRUE
  )
  
  
  ordered_y <- y01[
    ordering
  ]
  
  
  population_fraction <-
    
    seq_along(
      ordered_y
    ) /
    
    length(
      ordered_y
    )
  
  
  cumulative_positive <-
    
    cumsum(
      ordered_y
    ) /
    
    sum(
      ordered_y
    )
  
  
  lift <-
    
    cumulative_positive /
    
    population_fraction
  
  
  current_lift <- data.frame(
    
    Population_fraction =
      population_fraction,
    
    Lift =
      lift,
    
    Model =
      model_name
  )
  
  
  lift_data <- rbind(
    
    lift_data,
    
    current_lift
  )
}



write.csv(
  
  lift_data,
  
  file.path(
    result_dir,
    "15_test_lift_data.csv"
  ),
  
  row.names = FALSE
)



# ==============================================================================
# 50. Lift plot
# ==============================================================================

p_lift <- ggplot(
  
  lift_data,
  
  aes(
    
    x =
      Population_fraction,
    
    y =
      Lift,
    
    group =
      Model,
    
    colour =
      Model
  )
) +
  
  geom_hline(
    
    yintercept =
      1,
    
    linetype =
      2
  ) +
  
  geom_line() +
  
  labs(
    
    x =
      "Proportion of population targeted",
    
    y =
      "Lift",
    
    title =
      "Lift Curves in the Independent Test Set"
  ) +
  
  theme_classic()



ggsave(
  
  file.path(
    result_dir,
    "15_test_lift_curves.png"
  ),
  
  p_lift,
  
  width =
    8,
  
  height =
    7,
  
  dpi =
    600
)



# ==============================================================================
# 51. Kernel SHAP
#
# The stacking prediction wrapper still must not use type="prob"
# ==============================================================================

if (RUN_SHAP) {
  
  library(kernelshap)
  
  library(shapviz)
  
  
  set.seed(
    SEED + 20
  )
  
  
  background_n <- min(
    
    SHAP_BACKGROUND_N,
    
    nrow(
      dev_x
    )
  )
  
  
  background_index <- sample(
    
    seq_len(
      nrow(
        dev_x
      )
    ),
    
    size =
      background_n,
    
    replace =
      FALSE
  )
  
  
  background_data <- dev_x[
    background_index,
    ,
    drop = FALSE
  ]
  
  
  
  explain_n <- min(
    
    SHAP_EXPLAIN_N,
    
    nrow(
      test_x
    )
  )
  
  
  explain_index <- sample(
    
    seq_len(
      nrow(
        test_x
      )
    ),
    
    size =
      explain_n,
    
    replace =
      FALSE
  )
  
  
  explain_data <- test_x[
    explain_index,
    ,
    drop = FALSE
  ]
  
  
  
  stacking_prediction_function <- function(
    
    object,
    
    newdata
    
  ) {
    
    predict_stack_probability(
      
      object,
      
      newdata
    )
  }
  
  
  
  cat(
    "\n开始计算Stacking Kernel SHAP...\n"
  )
  
  
  set.seed(
    SEED + 21
  )
  
  
  kernel_shap_result <- kernelshap::kernelshap(
    
    object =
      stack_model,
    
    X =
      explain_data,
    
    bg_X =
      background_data,
    
    pred_fun =
      stacking_prediction_function,
    
    verbose =
      TRUE
  )
  
  
  
  shap_values <- as.data.frame(
    
    kernel_shap_result$S
  )
  
  
  write.csv(
    
    shap_values,
    
    file.path(
      result_dir,
      "16_Stacking_SHAP_values.csv"
    ),
    
    row.names = FALSE
  )
  
  
  
  shap_importance <- data.frame(
    
    feature =
      colnames(
        shap_values
      ),
    
    mean_abs_SHAP =
      colMeans(
        
        abs(
          shap_values
        ),
        
        na.rm =
          TRUE
      )
  )
  
  
  shap_importance <- shap_importance[
    
    order(
      -shap_importance$mean_abs_SHAP
    ),
    
    ,
    
    drop = FALSE
  ]
  
  
  write.csv(
    
    shap_importance,
    
    file.path(
      result_dir,
      "17_Stacking_SHAP_importance.csv"
    ),
    
    row.names = FALSE
  )
  
  
  
  cat(
    "\nSHAP Top 20：\n"
  )
  
  
  print(
    
    head(
      shap_importance,
      20
    )
  )
  
  
  
  shap_object <- shapviz::shapviz(
    
    kernel_shap_result
  )
  
  
  
  # SHAP importance
  p_shap_bar <- shapviz::sv_importance(
    
    shap_object,
    
    kind =
      "bar",
    
    max_display =
      min(
        20,
        length(
          rfe_features
        )
      )
  )
  
  
  ggsave(
    
    file.path(
      result_dir,
      "18_SHAP_importance_bar.png"
    ),
    
    p_shap_bar,
    
    width =
      8,
    
    height =
      7,
    
    dpi =
      600
  )
  
  
  
  # SHAP beeswarm
  p_shap_beeswarm <- shapviz::sv_importance(
    
    shap_object,
    
    kind =
      "beeswarm",
    
    max_display =
      min(
        20,
        length(
          rfe_features
        )
      )
  )
  
  
  ggsave(
    
    file.path(
      result_dir,
      "19_SHAP_beeswarm.png"
    ),
    
    p_shap_beeswarm,
    
    width =
      9,
    
    height =
      8,
    
    dpi =
      600
  )
  
  
  
  # Top 5 dependence plot
  top_features <- head(
    
    shap_importance$feature,
    
    5
  )
  
  
  for (
    current_feature in
    top_features
  ) {
    
    dependence_plot <- shapviz::sv_dependence(
      
      shap_object,
      
      v =
        current_feature
    )
    
    
    ggsave(
      
      file.path(
        
        result_dir,
        
        paste0(
          "SHAP_dependence_",
          current_feature,
          ".png"
        )
      ),
      
      dependence_plot,
      
      width =
        7,
      
      height =
        6,
      
      dpi =
        600
    )
  }
}



# ==============================================================================
# 52. Analysis configuration
# ==============================================================================

analysis_configuration <- data.frame(
  
  Item = c(
    
    "Candidate predictors",
    
    "Total N",
    
    "Development N",
    
    "Independent test N",
    
    "Development proportion",
    
    "Elastic Net alpha",
    
    "Elastic Net lambda rule",
    
    "Elastic Net selected",
    
    "RFE selected",
    
    "RFE folds",
    
    "RFE repeats",
    
    "ML folds",
    
    "ML repeats",
    
    "Imputation",
    
    "Scaling",
    
    "Resampling",
    
    "Stacking",
    
    "Stacking base probability",
    
    "Classification threshold",
    
    "SHAP"
  ),
  
  Value = c(
    
    70,
    
    nrow(
      ml_data
    ),
    
    nrow(
      development
    ),
    
    nrow(
      test
    ),
    
    TRAIN_PROP,
    
    best_alpha,
    
    EN_RULE,
    
    length(
      enet_features
    ),
    
    length(
      rfe_features
    ),
    
    RFE_FOLDS,
    
    RFE_REPEATS,
    
    ML_FOLDS,
    
    ML_REPEATS,
    
    "KNN imputation within resampling",
    
    "Center + scale within resampling",
    
    "SMOTE-Tomek in CV training folds only",
    
    "Logistic regression meta learner",
    
    "Probability of cognitive impairment (Yes)",
    
    CLASSIFICATION_THRESHOLD,
    
    ifelse(
      RUN_SHAP,
      "Kernel SHAP",
      "Not run"
    )
  )
)



write.csv(
  
  analysis_configuration,
  
  file.path(
    result_dir,
    "20_analysis_configuration.csv"
  ),
  
  row.names = FALSE
)



# ==============================================================================
# 53. Session info
# ==============================================================================

capture.output(
  
  sessionInfo(),
  
  file =
    file.path(
      result_dir,
      "21_sessionInfo.txt"
    )
)



# ==============================================================================
# 54. Final results
# ==============================================================================

cat(
  "\n============================================================\n"
)


cat(
  "机器学习分析完成\n"
)


cat(
  "============================================================\n"
)


cat(
  "总样本 =",
  nrow(
    ml_data
  ),
  "\n"
)


cat(
  "Development =",
  nrow(
    development
  ),
  "\n"
)


cat(
  "Independent Test =",
  nrow(
    test
  ),
  "\n"
)


cat(
  "\n初始候选变量 = 70\n"
)


cat(
  "Elastic Net =",
  length(
    enet_features
  ),
  "\n"
)


cat(
  "RFE final =",
  length(
    rfe_features
  ),
  "\n"
)


cat(
  "\nRFE最终特征：\n"
)


print(
  rfe_features
)


cat(
  "\nIndependent Test Performance：\n"
)


print(
  test_results
)


if (
  RUN_SHAP &&
  exists(
    "shap_importance"
  )
) {
  
  cat(
    "\nSHAP Top 20：\n"
  )
  
  
  print(
    
    head(
      shap_importance,
      20
    )
  )
}


cat(
  "\n所有结果保存至：\n"
)


cat(
  result_dir,
  "\n"
)


cat(
  "============================================================\n"
)
