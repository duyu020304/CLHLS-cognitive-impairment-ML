# ==============================================================================
# Exploratory indirect-pathway analysis
# Public-sharing version: Chinese comments translated to English.
# Analysis logic is unchanged; only comments and the local absolute project path were edited.
# Set data_dir to the appropriate local project directory before running.
# ==============================================================================

# ==============================================================================
# CLHLS 2018
# Exploratory mediation analysis based on SHAP-selected features
#
# Framework:
# Multiple upstream X + corresponding M + fixed Y
#
# Fixed outcome:
# Cognitive impairment
#
# Nine prespecified pathways:
#
# 1. Education -> Reading -> Cognitive impairment
# 2. Age -> Hearing impairment -> Cognitive impairment
# 3. Age -> Visual impairment -> Cognitive impairment
# 4. Visual impairment -> Reading -> Cognitive impairment
# 5. Education -> Nut products -> Cognitive impairment
# 6. Education -> Vegetable -> Cognitive impairment
# 7. Toothache -> Nut products -> Cognitive impairment
# 8. Toothache -> Vegetable -> Cognitive impairment
# 9. Age -> Disability risk -> Cognitive impairment
#
# IMPORTANT:
# - Age and Education are treated as continuous variables.
# - They are standardized.
# - One indirect-effect estimate corresponds to a 1-SD increase.
# - They are NOT divided into multiple categorical contrasts.
#
# Because this is a cross-sectional study:
# interpret results as exploratory indirect associations,
# NOT causal mediation.
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

SEED <- 20260921
SIMS <- 5000

set.seed(SEED)



# ==============================================================================
# 1. Packages
# ==============================================================================

packages <- c(
  "mediation",
  "dplyr",
  "broom",
  "ggplot2"
)

missing_packages <- packages[
  !vapply(
    packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_packages) > 0) {
  
  install.packages(
    missing_packages,
    dependencies = TRUE
  )
  
}

library(mediation)
library(dplyr)
library(broom)
library(ggplot2)



# ==============================================================================
# 2. File paths
# ==============================================================================

data_dir <- "path/to/project"

final_data_path <- file.path(
  data_dir,
  "CLHLS_70变量_最终分析样本.csv"
)

raw_data_path <- file.path(
  data_dir,
  "CLHLS_2018_cross.csv"
)

result_dir <- file.path(
  data_dir,
  "Mediation_SHAP_9_pathways"
)

if (!dir.exists(result_dir)) {
  
  dir.create(
    result_dir,
    recursive = TRUE
  )
  
}


if (!file.exists(final_data_path)) {
  
  stop(
    "找不到最终分析数据：",
    final_data_path
  )
  
}


if (!file.exists(raw_data_path)) {
  
  stop(
    "找不到原始CLHLS数据：",
    raw_data_path
  )
  
}



# ==============================================================================
# 3. Read final analysis sample
# ==============================================================================

dat <- read.csv(
  final_data_path,
  fileEncoding = "UTF-8-BOM",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

dat$id <- as.character(
  dat$id
)


cat(
  "\n============================================\n"
)

cat(
  "Final analysis sample N =",
  nrow(dat),
  "\n"
)

cat(
  "============================================\n"
)



# ==============================================================================
# 4. Restore ORIGINAL continuous education years from raw CLHLS
#
# f1 = years of schooling
#
# Your ML model may continue using grouped education.
# Mediation analysis uses original continuous years of schooling.
# ==============================================================================

raw <- read.csv(
  raw_data_path,
  fileEncoding = "GB18030",
  stringsAsFactors = FALSE,
  check.names = FALSE,
  strip.white = TRUE
)

raw$id <- as.character(
  raw$id
)


if (anyDuplicated(raw$id) > 0) {
  
  stop(
    "原始CLHLS数据中发现重复ID。"
  )
  
}


education_raw <- suppressWarnings(
  as.numeric(
    raw$f1
  )
)


# CLHLS:
# 88 = don't know
# 99 = missing
education_raw[
  education_raw %in%
    c(
      88,
      99
    )
] <- NA_real_


education_raw[
  education_raw < 0
] <- NA_real_


education_df <- data.frame(
  
  id =
    raw$id,
  
  education_years =
    education_raw,
  
  stringsAsFactors =
    FALSE
  
)



# ==============================================================================
# 5. Merge continuous education into final sample
# ==============================================================================

idx <- match(
  dat$id,
  education_df$id
)


if (any(is.na(idx))) {
  
  warning(
    sum(is.na(idx)),
    "名最终样本没有匹配到原始f1教育变量。"
  )
  
}


dat$education_years <-
  education_df$education_years[
    idx
  ]


cat(
  "\nEducation years distribution:\n"
)

print(
  summary(
    dat$education_years
  )
)



# ==============================================================================
# 6. Required variables
# ==============================================================================

required_variables <- c(
  
  "cognitive_impairment",
  
  "age_continuous",
  
  "education_years",
  
  "gender",
  
  "nation",
  
  "read",
  
  "hearing_impairment",
  
  "visual_impairment",
  
  "nut_products",
  
  "vegetable",
  
  "toothache",
  
  "disability_risk"
  
)


missing_variables <- setdiff(
  required_variables,
  names(dat)
)


if (length(missing_variables) > 0) {
  
  stop(
    "缺少以下变量：",
    paste(
      missing_variables,
      collapse = ", "
    )
  )
  
}



# ==============================================================================
# 7. Convert numeric variables
# ==============================================================================

for (v in required_variables) {
  
  dat[[v]] <- suppressWarnings(
    as.numeric(
      dat[[v]]
    )
  )
  
}



# ==============================================================================
# 8. Define outcome / X / M variables
# ==============================================================================

dat$cognitive_bin <-
  dat$cognitive_impairment


dat$read_bin <-
  dat$read


dat$hearing_bin <-
  dat$hearing_impairment


dat$visual_bin <-
  dat$visual_impairment


dat$nut_bin <-
  dat$nut_products


dat$vegetable_bin <-
  dat$vegetable


dat$toothache_bin <-
  dat$toothache


dat$disability_bin <-
  dat$disability_risk



# ==============================================================================
# 9. Standardize continuous upstream exposures
#
# Age and Education remain ONE continuous variable.
#
# A mediation result corresponds to:
#
# X = -0.5 SD -> +0.5 SD
#
# therefore exactly a 1-SD difference.
# ==============================================================================

age_mean <- mean(
  dat$age_continuous,
  na.rm = TRUE
)

age_sd <- sd(
  dat$age_continuous,
  na.rm = TRUE
)


education_mean <- mean(
  dat$education_years,
  na.rm = TRUE
)

education_sd <- sd(
  dat$education_years,
  na.rm = TRUE
)


if (
  !is.finite(age_sd) ||
  age_sd <= 0
) {
  
  stop(
    "age_continuous没有有效变异。"
  )
  
}


if (
  !is.finite(education_sd) ||
  education_sd <= 0
) {
  
  stop(
    "education_years没有有效变异。"
  )
  
}


dat$age_std <- (
  dat$age_continuous -
    age_mean
) / age_sd


dat$education_std <- (
  dat$education_years -
    education_mean
) / education_sd



cat(
  "\n============================================\n"
)

cat(
  "Age mean =",
  round(age_mean, 2),
  "; SD =",
  round(age_sd, 2),
  "\n"
)

cat(
  "Education mean =",
  round(education_mean, 2),
  "; SD =",
  round(education_sd, 2),
  " years\n"
)

cat(
  "============================================\n"
)



# ==============================================================================
# 10. Basic confounders
# ==============================================================================

dat$gender_factor <- factor(
  dat$gender
)

dat$nation_factor <- factor(
  dat$nation
)



# ==============================================================================
# 11. Validate binary variables
# ==============================================================================

binary_variables <- c(
  
  "cognitive_bin",
  
  "read_bin",
  
  "hearing_bin",
  
  "visual_bin",
  
  "nut_bin",
  
  "vegetable_bin",
  
  "toothache_bin",
  
  "disability_bin"
  
)


for (v in binary_variables) {
  
  observed_values <- sort(
    unique(
      na.omit(
        dat[[v]]
      )
    )
  )
  
  
  if (
    !all(
      observed_values %in%
      c(
        0,
        1
      )
    )
  ) {
    
    stop(
      v,
      "不是0/1变量。实际值：",
      paste(
        observed_values,
        collapse = ", "
      )
    )
    
  }
  
}



# ==============================================================================
# 12. Save coding check
# ==============================================================================

coding_check <- list(
  
  outcome =
    table(
      dat$cognitive_bin,
      useNA = "ifany"
    ),
  
  read =
    table(
      dat$read_bin,
      useNA = "ifany"
    ),
  
  hearing =
    table(
      dat$hearing_bin,
      useNA = "ifany"
    ),
  
  visual =
    table(
      dat$visual_bin,
      useNA = "ifany"
    ),
  
  nuts =
    table(
      dat$nut_bin,
      useNA = "ifany"
    ),
  
  vegetable =
    table(
      dat$vegetable_bin,
      useNA = "ifany"
    ),
  
  toothache =
    table(
      dat$toothache_bin,
      useNA = "ifany"
    ),
  
  disability =
    table(
      dat$disability_bin,
      useNA = "ifany"
    ),
  
  age =
    summary(
      dat$age_continuous
    ),
  
  education =
    summary(
      dat$education_years
    )
  
)


capture.output(
  
  coding_check,
  
  file =
    file.path(
      result_dir,
      "00_Variable_coding_check.txt"
    )
  
)



# ==============================================================================
# 13. General mediation function
#
# M = binary
# Y = binary
#
# Continuous X:
#   standardized, one result = 1-SD difference
#
# Binary X:
#   0 -> 1
#
# Primary indirect effect:
#   Average ACME from mediation::mediate()
#
# Do NOT use simple a*b as the formal indirect effect,
# because mediator and outcome are logistic models.
# ==============================================================================

run_mediation_pathway <- function(
    
  data,
  
  exposure,
  
  mediator,
  
  covariates,
  
  control_value,
  
  treatment_value,
  
  pathway_name,
  
  pathway_level,
  
  theoretical_basis,
  
  sims = 5000
  
) {
  
  
  # ---------------------------------------------------------------------------
  # Model-specific complete-case dataset
  # ---------------------------------------------------------------------------
  
  needed_variables <- unique(
    c(
      exposure,
      mediator,
      "cognitive_bin",
      covariates
    )
  )
  
  
  d <- data[
    ,
    needed_variables,
    drop = FALSE
  ]
  
  
  d <- d[
    complete.cases(
      d
    ),
    ,
    drop = FALSE
  ]
  
  
  d <- droplevels(
    d
  )
  
  
  N <- nrow(
    d
  )
  
  
  if (N == 0) {
    
    stop(
      pathway_name,
      ": N=0."
    )
    
  }
  
  
  
  # ---------------------------------------------------------------------------
  # Mediator check
  # ---------------------------------------------------------------------------
  
  mediator_values <- sort(
    unique(
      d[[mediator]]
    )
  )
  
  
  if (
    !all(
      mediator_values %in%
      c(
        0,
        1
      )
    )
  ) {
    
    stop(
      pathway_name,
      ": mediator不是0/1变量。"
    )
    
  }
  
  
  
  # ---------------------------------------------------------------------------
  # Console information
  # ---------------------------------------------------------------------------
  
  cat(
    "\n============================================================\n"
  )
  
  cat(
    "PATHWAY:\n",
    pathway_name,
    "\n"
  )
  
  cat(
    "Level:",
    pathway_level,
    "\n"
  )
  
  cat(
    "N =",
    N,
    "\n"
  )
  
  cat(
    "Exposure:",
    exposure,
    "\n"
  )
  
  cat(
    "Mediator:",
    mediator,
    "\n"
  )
  
  
  
  # ===========================================================================
  # A. TOTAL-EFFECT MODEL
  #
  # Y ~ X + confounders
  # ===========================================================================
  
  total_formula <- reformulate(
    
    termlabels =
      c(
        exposure,
        covariates
      ),
    
    response =
      "cognitive_bin"
    
  )
  
  
  total_model <- glm(
    
    formula =
      total_formula,
    
    data =
      d,
    
    family =
      binomial(
        link = "logit"
      )
    
  )
  
  
  total_model$call$formula <-
    total_formula
  
  
  
  # ===========================================================================
  # B. MEDIATOR MODEL
  #
  # M ~ X + confounders
  # ===========================================================================
  
  mediator_formula <- reformulate(
    
    termlabels =
      c(
        exposure,
        covariates
      ),
    
    response =
      mediator
    
  )
  
  
  mediator_model <- glm(
    
    formula =
      mediator_formula,
    
    data =
      d,
    
    family =
      binomial(
        link = "logit"
      )
    
  )
  
  
  mediator_model$call$formula <-
    mediator_formula
  
  
  
  # ===========================================================================
  # C. OUTCOME MODEL
  #
  # Y ~ X + M + confounders
  # ===========================================================================
  
  outcome_formula <- reformulate(
    
    termlabels =
      c(
        exposure,
        mediator,
        covariates
      ),
    
    response =
      "cognitive_bin"
    
  )
  
  
  outcome_model <- glm(
    
    formula =
      outcome_formula,
    
    data =
      d,
    
    family =
      binomial(
        link = "logit"
      )
    
  )
  
  
  outcome_model$call$formula <-
    outcome_formula
  
  
  
  # ===========================================================================
  # Check convergence
  # ===========================================================================
  
  if (!total_model$converged) {
    
    warning(
      pathway_name,
      ": total model did not converge."
    )
    
  }
  
  
  if (!mediator_model$converged) {
    
    warning(
      pathway_name,
      ": mediator model did not converge."
    )
    
  }
  
  
  if (!outcome_model$converged) {
    
    warning(
      pathway_name,
      ": outcome model did not converge."
    )
    
  }
  
  
  
  # ===========================================================================
  # Extract a / b / c / c'
  #
  # NOTE:
  # These are logistic-regression coefficients.
  # ===========================================================================
  
  a_coef <- unname(
    coef(
      mediator_model
    )[exposure]
  )
  
  
  b_coef <- unname(
    coef(
      outcome_model
    )[mediator]
  )
  
  
  c_coef <- unname(
    coef(
      total_model
    )[exposure]
  )
  
  
  cprime_coef <- unname(
    coef(
      outcome_model
    )[exposure]
  )
  
  
  
  # ===========================================================================
  # Formal bootstrap mediation
  # ===========================================================================
  
  set.seed(
    SEED
  )
  
  
  mediation_result <- mediation::mediate(
    
    model.m =
      mediator_model,
    
    model.y =
      outcome_model,
    
    treat =
      exposure,
    
    mediator =
      mediator,
    
    control.value =
      control_value,
    
    treat.value =
      treatment_value,
    
    boot =
      TRUE,
    
    boot.ci.type =
      "perc",
    
    sims =
      sims,
    
    conf.level =
      0.95,
    
    dropobs =
      FALSE
    
  )
  
  
  
  # ===========================================================================
  # Save individual detailed results
  # ===========================================================================
  
  safe_name <- gsub(
    "[^A-Za-z0-9]+",
    "_",
    pathway_name
  )
  
  
  write.csv(
    
    broom::tidy(
      mediator_model,
      exponentiate = TRUE,
      conf.int = TRUE
    ),
    
    file.path(
      result_dir,
      paste0(
        safe_name,
        "_Mediator_model_OR.csv"
      )
    ),
    
    row.names = FALSE
    
  )
  
  
  write.csv(
    
    broom::tidy(
      outcome_model,
      exponentiate = TRUE,
      conf.int = TRUE
    ),
    
    file.path(
      result_dir,
      paste0(
        safe_name,
        "_Outcome_model_OR.csv"
      )
    ),
    
    row.names = FALSE
    
  )
  
  
  capture.output(
    
    summary(
      mediation_result
    ),
    
    file =
      file.path(
        result_dir,
        paste0(
          safe_name,
          "_Full_mediation_result.txt"
        )
      )
    
  )
  
  
  
  # ===========================================================================
  # Extract main results
  # ===========================================================================
  
  result <- data.frame(
    
    Pathway =
      pathway_name,
    
    Level =
      pathway_level,
    
    Theoretical_basis =
      theoretical_basis,
    
    N =
      N,
    
    Exposure =
      exposure,
    
    Mediator =
      mediator,
    
    
    # -------------------------------------------------------------------------
    # Formal mediation estimates
    # -------------------------------------------------------------------------
    
    Total_effect =
      mediation_result$tau.coef,
    
    Indirect_effect =
      mediation_result$d.avg,
    
    Direct_effect =
      mediation_result$z.avg,
    
    Proportion_mediated =
      mediation_result$n.avg,
    
    
    Indirect_CI_low =
      mediation_result$d.avg.ci[1],
    
    Indirect_CI_high =
      mediation_result$d.avg.ci[2],
    
    Indirect_P =
      mediation_result$d.avg.p,
    
    
    Direct_CI_low =
      mediation_result$z.avg.ci[1],
    
    Direct_CI_high =
      mediation_result$z.avg.ci[2],
    
    Direct_P =
      mediation_result$z.avg.p,
    
    
    Total_CI_low =
      mediation_result$tau.ci[1],
    
    Total_CI_high =
      mediation_result$tau.ci[2],
    
    Total_P =
      mediation_result$tau.p,
    
    
    # -------------------------------------------------------------------------
    # Traditional path coefficients
    # -------------------------------------------------------------------------
    
    a_logit =
      a_coef,
    
    b_logit =
      b_coef,
    
    c_logit =
      c_coef,
    
    c_prime_logit =
      cprime_coef,
    
    stringsAsFactors =
      FALSE
    
  )
  
  
  
  # ---------------------------------------------------------------------------
  # Derived variables
  # ---------------------------------------------------------------------------
  
  result$Proportion_percent <-
    result$Proportion_mediated *
    100
  
  
  result$Indirect_CI_excludes_zero <- (
    result$Indirect_CI_low > 0 &
      result$Indirect_CI_high > 0
  ) |
    (
      result$Indirect_CI_low < 0 &
        result$Indirect_CI_high < 0
    )
  
  
  return(
    result
  )
  
}



# ==============================================================================
# 14. Define nine prespecified pathways
#
# NOTE:
#
# X and M are restricted to SHAP-retained important features.
#
# Confounders do not need to be SHAP-selected because their role is adjustment,
# not prediction/pathway selection.
# ==============================================================================


pathway_specs <- list(
  
  
  # ============================================================================
  # 1. EDUCATION -> READING
  # ============================================================================
  
  list(
    
    exposure =
      "education_std",
    
    mediator =
      "read_bin",
    
    covariates =
      c(
        "age_std",
        "gender_factor",
        "nation_factor"
      ),
    
    control =
      -0.5,
    
    treatment =
      0.5,
    
    pathway =
      "Education -> Reading -> Cognitive impairment",
    
    level =
      "Primary",
    
    theory =
      "Cognitive reserve / cognitive stimulation"
    
  ),
  
  
  
  # ============================================================================
  # 2. AGE -> HEARING IMPAIRMENT
  # ============================================================================
  
  list(
    
    exposure =
      "age_std",
    
    mediator =
      "hearing_bin",
    
    covariates =
      c(
        "gender_factor",
        "nation_factor",
        "education_std"
      ),
    
    control =
      -0.5,
    
    treatment =
      0.5,
    
    pathway =
      "Age -> Hearing impairment -> Cognitive impairment",
    
    level =
      "Primary",
    
    theory =
      "Age-related sensory decline / cognitive load"
    
  ),
  
  
  
  # ============================================================================
  # 3. AGE -> VISUAL IMPAIRMENT
  # ============================================================================
  
  list(
    
    exposure =
      "age_std",
    
    mediator =
      "visual_bin",
    
    covariates =
      c(
        "gender_factor",
        "nation_factor",
        "education_std"
      ),
    
    control =
      -0.5,
    
    treatment =
      0.5,
    
    pathway =
      "Age -> Visual impairment -> Cognitive impairment",
    
    level =
      "Primary",
    
    theory =
      "Age-related sensory decline"
    
  ),
  
  
  
  # ============================================================================
  # 4. VISUAL IMPAIRMENT -> READING
  # ============================================================================
  
  list(
    
    exposure =
      "visual_bin",
    
    mediator =
      "read_bin",
    
    covariates =
      c(
        "age_std",
        "gender_factor",
        "nation_factor",
        "education_std"
      ),
    
    control =
      0,
    
    treatment =
      1,
    
    pathway =
      "Visual impairment -> Reading -> Cognitive impairment",
    
    level =
      "Primary",
    
    theory =
      "Sensory limitation -> reduced cognitive stimulation"
    
  ),
  
  
  
  # ============================================================================
  # 5. EDUCATION -> NUT PRODUCTS
  # ============================================================================
  
  list(
    
    exposure =
      "education_std",
    
    mediator =
      "nut_bin",
    
    covariates =
      c(
        "age_std",
        "gender_factor",
        "nation_factor"
      ),
    
    control =
      -0.5,
    
    treatment =
      0.5,
    
    pathway =
      "Education -> Nut products -> Cognitive impairment",
    
    level =
      "Secondary",
    
    theory =
      "Education -> dietary behavior -> cognition"
    
  ),
  
  
  
  # ============================================================================
  # 6. EDUCATION -> VEGETABLE
  # ============================================================================
  
  list(
    
    exposure =
      "education_std",
    
    mediator =
      "vegetable_bin",
    
    covariates =
      c(
        "age_std",
        "gender_factor",
        "nation_factor"
      ),
    
    control =
      -0.5,
    
    treatment =
      0.5,
    
    pathway =
      "Education -> Vegetable intake -> Cognitive impairment",
    
    level =
      "Secondary",
    
    theory =
      "Education -> dietary behavior -> cognition"
    
  ),
  
  
  
  # ============================================================================
  # 7. TOOTHACHE -> NUT PRODUCTS
  # ============================================================================
  
  list(
    
    exposure =
      "toothache_bin",
    
    mediator =
      "nut_bin",
    
    covariates =
      c(
        "age_std",
        "gender_factor",
        "nation_factor",
        "education_std"
      ),
    
    control =
      0,
    
    treatment =
      1,
    
    pathway =
      "Toothache -> Nut products -> Cognitive impairment",
    
    level =
      "Secondary",
    
    theory =
      "Oral pain -> dietary restriction -> cognition"
    
  ),
  
  
  
  # ============================================================================
  # 8. TOOTHACHE -> VEGETABLE
  # ============================================================================
  
  list(
    
    exposure =
      "toothache_bin",
    
    mediator =
      "vegetable_bin",
    
    covariates =
      c(
        "age_std",
        "gender_factor",
        "nation_factor",
        "education_std"
      ),
    
    control =
      0,
    
    treatment =
      1,
    
    pathway =
      "Toothache -> Vegetable intake -> Cognitive impairment",
    
    level =
      "Secondary",
    
    theory =
      "Oral health -> diet/nutrition -> cognition"
    
  ),
  
  
  
  # ============================================================================
  # 9. AGE -> DISABILITY
  #
  # Important caveat:
  # cognition may also contribute to ADL disability.
  #
  # Therefore this should remain a secondary exploratory pathway.
  # ============================================================================
  
  list(
    
    exposure =
      "age_std",
    
    mediator =
      "disability_bin",
    
    covariates =
      c(
        "gender_factor",
        "nation_factor",
        "education_std"
      ),
    
    control =
      -0.5,
    
    treatment =
      0.5,
    
    pathway =
      "Age -> Disability risk -> Cognitive impairment",
    
    level =
      "Secondary",
    
    theory =
      "Age-related functional decline; reverse causality remains possible"
    
  )
  
)



# ==============================================================================
# 15. Run all nine pathways
# ==============================================================================

results_list <- vector(
  "list",
  length(
    pathway_specs
  )
)


for (i in seq_along(pathway_specs)) {
  
  
  spec <- pathway_specs[[i]]
  
  
  results_list[[i]] <- run_mediation_pathway(
    
    data =
      dat,
    
    exposure =
      spec$exposure,
    
    mediator =
      spec$mediator,
    
    covariates =
      spec$covariates,
    
    control_value =
      spec$control,
    
    treatment_value =
      spec$treatment,
    
    pathway_name =
      spec$pathway,
    
    pathway_level =
      spec$level,
    
    theoretical_basis =
      spec$theory,
    
    sims =
      SIMS
    
  )
  
}



# ==============================================================================
# 16. Combine results
# ==============================================================================

all_results <- bind_rows(
  results_list
)



# ==============================================================================
# 17. FDR correction
#
# Correct across ALL 9 prespecified indirect-effect tests.
# ==============================================================================

all_results$Indirect_P_FDR <- p.adjust(
  
  all_results$Indirect_P,
  
  method =
    "BH"
  
)



# ==============================================================================
# 18. Define statistically supported pathways
#
# Both conditions:
#
# 1. bootstrap 95% CI does not include zero
# 2. BH-FDR P < 0.05
# ==============================================================================

all_results$Supported <- with(
  
  all_results,
  
  Indirect_CI_excludes_zero &
    Indirect_P_FDR < 0.05
  
)



# ==============================================================================
# 19. Formatted CI
# ==============================================================================

all_results$Indirect_95CI <- sprintf(
  
  "%.4f to %.4f",
  
  all_results$Indirect_CI_low,
  
  all_results$Indirect_CI_high
  
)



# ==============================================================================
# 20. Save ALL nine pathways
#
# Recommended for Supplementary Material.
# ==============================================================================

write.csv(
  
  all_results,
  
  file.path(
    result_dir,
    "01_ALL_9_mediation_pathways.csv"
  ),
  
  row.names =
    FALSE
  
)



# ==============================================================================
# 21. Supported pathways only
#
# Candidate table for main manuscript.
# ==============================================================================

supported_results <- all_results %>%
  
  filter(
    Supported
  )


write.csv(
  
  supported_results,
  
  file.path(
    result_dir,
    "02_FDR_supported_pathways.csv"
  ),
  
  row.names =
    FALSE
  
)



# ==============================================================================
# 22. Manuscript-ready table
#
# Similar structure to your previous mediation table.
#
# Note:
# Indirect effect is ACME from mediation package,
# NOT simply a*b.
# ==============================================================================

main_table <- supported_results %>%
  
  transmute(
    
    Item =
      Pathway,
    
    N =
      N,
    
    `c Total effect` =
      Total_effect,
    
    `Indirect effect` =
      Indirect_effect,
    
    `c' Direct effect` =
      Direct_effect,
    
    `Proportion of effect (%)` =
      Proportion_percent,
    
    `Indirect effect (95% BootCI)` =
      Indirect_95CI,
    
    `a` =
      a_logit,
    
    `b` =
      b_logit,
    
    `Raw P` =
      Indirect_P,
    
    `BH-FDR P` =
      Indirect_P_FDR
    
  )


write.csv(
  
  main_table,
  
  file.path(
    result_dir,
    "03_Main_manuscript_mediation_table.csv"
  ),
  
  row.names =
    FALSE
  
)



# ==============================================================================
# 23. Supplementary table
# ==============================================================================

supplement_table <- all_results %>%
  
  transmute(
    
    Pathway =
      Pathway,
    
    Level =
      Level,
    
    Theoretical_basis =
      Theoretical_basis,
    
    N =
      N,
    
    Total_effect =
      Total_effect,
    
    Indirect_effect =
      Indirect_effect,
    
    Direct_effect =
      Direct_effect,
    
    Proportion_percent =
      Proportion_percent,
    
    BootCI =
      Indirect_95CI,
    
    Raw_P =
      Indirect_P,
    
    FDR_P =
      Indirect_P_FDR,
    
    Supported =
      Supported
    
  )


write.csv(
  
  supplement_table,
  
  file.path(
    result_dir,
    "04_Supplement_ALL_9_pathways.csv"
  ),
  
  row.names =
    FALSE
  
)



# ==============================================================================
# 24. Save pathway registry
# ==============================================================================

registry <- bind_rows(
  
  lapply(
    
    pathway_specs,
    
    function(x) {
      
      data.frame(
        
        Pathway =
          x$pathway,
        
        Level =
          x$level,
        
        Theoretical_basis =
          x$theory,
        
        Exposure =
          x$exposure,
        
        Mediator =
          x$mediator,
        
        Covariates =
          paste(
            x$covariates,
            collapse = "; "
          ),
        
        stringsAsFactors =
          FALSE
        
      )
      
    }
    
  )
  
)


write.csv(
  
  registry,
  
  file.path(
    result_dir,
    "05_Pathway_registry.csv"
  ),
  
  row.names =
    FALSE
  
)



# ==============================================================================
# 25. Save continuous-variable interpretation
# ==============================================================================

continuous_info <- data.frame(
  
  Variable =
    c(
      "Age",
      "Education"
    ),
  
  Original_measure =
    c(
      "Validated age in years",
      "Years of schooling (CLHLS f1)"
    ),
  
  Mean =
    c(
      age_mean,
      education_mean
    ),
  
  SD =
    c(
      age_sd,
      education_sd
    ),
  
  Effect_interpretation =
    c(
      "Effect corresponding to a 1-SD increase in age",
      "Effect corresponding to a 1-SD increase in years of schooling"
    ),
  
  stringsAsFactors =
    FALSE
  
)


write.csv(
  
  continuous_info,
  
  file.path(
    result_dir,
    "06_Age_Education_effect_definition.csv"
  ),
  
  row.names =
    FALSE
  
)



# ==============================================================================
# 26. Forest plot - all nine pathways
# ==============================================================================

plot_data <- all_results


plot_data$Pathway <- factor(
  
  plot_data$Pathway,
  
  levels =
    rev(
      plot_data$Pathway
    )
  
)


p_all <- ggplot(
  
  plot_data,
  
  aes(
    x =
      Indirect_effect,
    
    y =
      Pathway
  )
  
) +
  
  geom_vline(
    xintercept =
      0,
    
    linetype =
      2
  ) +
  
  geom_point(
    size =
      2.5
  ) +
  
  geom_errorbar(
    
    aes(
      xmin =
        Indirect_CI_low,
      
      xmax =
        Indirect_CI_high
    ),
    
    orientation =
      "y",
    
    width =
      0.15
    
  ) +
  
  labs(
    
    x =
      "Average indirect effect (95% bootstrap CI)",
    
    y =
      NULL,
    
    title =
      "Exploratory indirect pathways based on SHAP-selected features"
    
  ) +
  
  theme_classic()



ggsave(
  
  file.path(
    result_dir,
    "07_ALL_9_pathways_forest_plot.png"
  ),
  
  p_all,
  
  width =
    11,
  
  height =
    8,
  
  dpi =
    600
  
)



# ==============================================================================
# 27. Forest plot - supported pathways only
# ==============================================================================

if (nrow(supported_results) > 0) {
  
  
  plot_supported <- supported_results
  
  
  plot_supported$Pathway <- factor(
    
    plot_supported$Pathway,
    
    levels =
      rev(
        plot_supported$Pathway
      )
    
  )
  
  
  p_supported <- ggplot(
    
    plot_supported,
    
    aes(
      x =
        Indirect_effect,
      
      y =
        Pathway
    )
    
  ) +
    
    geom_vline(
      xintercept =
        0,
      
      linetype =
        2
    ) +
    
    geom_point(
      size =
        2.7
    ) +
    
    geom_errorbar(
      
      aes(
        xmin =
          Indirect_CI_low,
        
        xmax =
          Indirect_CI_high
      ),
      
      orientation =
        "y",
      
      width =
        0.15
      
    ) +
    
    labs(
      
      x =
        "Average indirect effect (95% bootstrap CI)",
      
      y =
        NULL,
      
      title =
        "FDR-supported exploratory indirect pathways"
      
    ) +
    
    theme_classic()
  
  
  ggsave(
    
    file.path(
      result_dir,
      "08_FDR_supported_pathways_forest_plot.png"
    ),
    
    p_supported,
    
    width =
      11,
    
    height =
      max(
        5,
        0.7 *
          nrow(
            supported_results
          )
      ),
    
    dpi =
      600
    
  )
  
}



# ==============================================================================
# 28. Session info
# ==============================================================================

capture.output(
  
  sessionInfo(),
  
  file =
    file.path(
      result_dir,
      "09_sessionInfo.txt"
    )
  
)



# ==============================================================================
# 29. Final console output
# ==============================================================================

cat(
  "\n============================================================\n"
)

cat(
  "All nine mediation analyses completed.\n"
)

cat(
  "Total prespecified pathways =",
  nrow(
    all_results
  ),
  "\n"
)

cat(
  "FDR-supported pathways =",
  nrow(
    supported_results
  ),
  "\n"
)

cat(
  "\nFull results:\n"
)

cat(
  file.path(
    result_dir,
    "01_ALL_9_mediation_pathways.csv"
  ),
  "\n"
)

cat(
  "\nMain manuscript table:\n"
)

cat(
  file.path(
    result_dir,
    "03_Main_manuscript_mediation_table.csv"
  ),
  "\n"
)

cat(
  "\nSupplementary table:\n"
)

cat(
  file.path(
    result_dir,
    "04_Supplement_ALL_9_pathways.csv"
  ),
  "\n"
)

cat(
  "============================================================\n"
)


print(
  main_table
)
