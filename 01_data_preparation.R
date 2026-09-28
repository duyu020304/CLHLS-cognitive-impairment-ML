# ==============================================================================
# Data preparation, candidate-predictor construction, and sample selection
# Public-sharing version: Chinese comments translated to English.
# Analysis logic is unchanged; only comments and the local absolute project path were edited.
# Set data_dir to the appropriate local project directory before running.
# ==============================================================================

# ==============================================================================
# CLHLS 2018
# 70 candidate predictors + final machine-learning analysis sample
#
# Six variables removed from the original 76 candidate predictors:
#
# 1. dietary_diversity
#    -> Derived from 13 food variables; remove the composite variable and retain the original food variables
#
# 2. ventilation_frequency
#    -> Derived from four seasonal ventilation variables; remove the composite variable and retain the seasonal variables
#
# 3. dementia
#    -> Too closely related to the current cognitive-impairment outcome; potential target leakage
#
# 4. chronic_disease_comorbidity
#    -> Derived from individual chronic-disease variables, and the original definition included dementia
#
# 5. breast_hyperplasia
#    -> Excluded
#
# 6. uterine_tumor
#    -> Excluded
#
# Final candidate predictors: 70
#
# Final sample eligibility criteria:
# 1. validated age >=65 years
# 2. all seven GAD-7 items are scoreable and total score >=4
# 3. CMMSE can be fully scored
#
# IMPORTANT:
# - Do not exclude participants because of missingness among the 70 candidate predictors
# - Do not impute missing values at this stage
# - Do not standardize at this stage
# - Do not apply SMOTE at this stage
# - Do not perform univariable P-value screening
# - Do not perform multivariable P-value screening
# - Do not perform LASSO
#
# Missing-value imputation, dummy-variable encoding, standardization, feature selection, SMOTE, etc.
# are deferred to the subsequent machine-learning training/CV pipeline.
#
# Expected counts based on the current raw data:
#
# Original sample                 15874
# Age >=65                 15771
# GAD-7>=4                  2242
# CMMSE scoreable               2081
#
# Final interview-year distribution:
# 2017 = 158
# 2018 = 1554
# 2019 = 369
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



# ==============================================================================
# 1. File paths
# ==============================================================================

data_dir <- "path/to/project"


csv_path <- file.path(
  data_dir,
  "CLHLS_2018_cross.csv"
)


out_all <- file.path(
  data_dir,
  "CLHLS_70变量_全部样本_编码后.csv"
)


out_final <- file.path(
  data_dir,
  "CLHLS_70变量_最终分析样本.csv"
)


out_flow <- file.path(
  data_dir,
  "CLHLS_70变量_样本筛选流程.csv"
)


out_qc <- file.path(
  data_dir,
  "CLHLS_70变量_缺失与值域质控.csv"
)


out_year <- file.path(
  data_dir,
  "CLHLS_70变量_年份分布.csv"
)


out_dict <- file.path(
  data_dir,
  "CLHLS_70变量_字段字典.csv"
)


if (!dir.exists(data_dir)) {

  stop(
    "文件夹不存在：",
    data_dir
  )
}


if (!file.exists(csv_path)) {

  stop(
    "找不到原始CSV文件：",
    csv_path
  )
}



# ==============================================================================
# 2. Helper functions
# ==============================================================================


# ------------------------------------------------------------------------------
# Convert to numeric
# ------------------------------------------------------------------------------

to_num <- function(x) {

  y <- trimws(
    as.character(x)
  )

  y[y == ""] <- NA_character_

  suppressWarnings(
    as.numeric(y)
  )
}



# ------------------------------------------------------------------------------
# Recode according to mapping
#
# Values not included in the mapping are set to NA
# ------------------------------------------------------------------------------

recode_num <- function(x, mapping) {

  x_num <- to_num(x)

  old_values <- as.numeric(
    names(mapping)
  )

  idx <- match(
    x_num,
    old_values
  )

  out <- rep(
    NA_real_,
    length(x_num)
  )

  valid <- !is.na(idx)

  out[valid] <- as.numeric(
    unname(
      mapping[idx[valid]]
    )
  )

  out
}



# ------------------------------------------------------------------------------
# Common Yes/No disease or behavior variables
#
# 1 = yes
# 2 = no
# 8 = don't know
# 9 = missing
#
# Output:
# yes = 1
# no  = 0
# ------------------------------------------------------------------------------

yes_no <- function(x) {

  recode_num(
    x,
    c(
      "1" = 1,
      "2" = 0
    )
  )
}



# ------------------------------------------------------------------------------
# Psychological frequency items
#
# Original:
# 1 always
# 2 often
# 3 sometimes
# 4 seldom
# 5 never
#
# Conversion:
# 3 / 2 / 1 / 1 / 0
# ------------------------------------------------------------------------------

freq_0_3 <- function(x) {

  recode_num(
    x,
    c(
      "1" = 3,
      "2" = 2,
      "3" = 1,
      "4" = 1,
      "5" = 0
    )
  )
}



# ------------------------------------------------------------------------------
# CES-D negative items
# ------------------------------------------------------------------------------

cesd_negative_item <- function(x) {

  recode_num(
    x,
    c(
      "1" = 3,
      "2" = 2,
      "3" = 1,
      "4" = 1,
      "5" = 0
    )
  )
}



# ------------------------------------------------------------------------------
# Reverse-score positive CES-D items
# ------------------------------------------------------------------------------

cesd_positive_item <- function(x) {

  recode_num(
    x,
    c(
      "1" = 0,
      "2" = 1,
      "3" = 2,
      "4" = 2,
      "5" = 3
    )
  )
}



# ------------------------------------------------------------------------------
# Fruit and vegetables
# ------------------------------------------------------------------------------

food_fruit_veg <- function(x) {

  recode_num(
    x,
    c(
      "1" = 1,
      "2" = 1,
      "3" = 0,
      "4" = 0
    )
  )
}



# ------------------------------------------------------------------------------
# Other food variables
# ------------------------------------------------------------------------------

food_other <- function(x) {

  recode_num(
    x,
    c(
      "1" = 1,
      "2" = 1,
      "3" = 0,
      "4" = 0,
      "5" = 0
    )
  )
}



# ------------------------------------------------------------------------------
# Calculate the total score only when all component items are scoreable
# ------------------------------------------------------------------------------

row_sum_complete <- function(x) {

  x <- as.matrix(x)

  storage.mode(x) <- "numeric"

  score <- rowSums(
    x,
    na.rm = TRUE
  )

  score[
    rowSums(
      is.na(x)
    ) > 0
  ] <- NA_real_

  score
}



# ------------------------------------------------------------------------------
# ADL：
#
# Any difficulty in an ADL item -> 1
# All items explicitly indicate no difficulty -> 0
# No positive item, but at least one item is missing -> NA
# ------------------------------------------------------------------------------

row_any_positive <- function(x) {

  x <- as.matrix(x)

  storage.mode(x) <- "numeric"

  positive_n <- rowSums(
    x == 1,
    na.rm = TRUE
  )

  missing_n <- rowSums(
    is.na(x)
  )

  ifelse(
    positive_n > 0,
    1,
    ifelse(
      missing_n > 0,
      NA_real_,
      0
    )
  )
}



# ------------------------------------------------------------------------------
# Export CSV
#
# When readr is installed, write with a UTF-8 BOM automatically,
# which improves compatibility with Chinese text in Excel.
# ------------------------------------------------------------------------------

write_csv_blank <- function(x, path) {

  if (
    requireNamespace(
      "readr",
      quietly = TRUE
    )
  ) {

    readr::write_excel_csv(
      x,
      file = path,
      na = ""
    )

  } else {

    utils::write.csv(
      x,
      file = path,
      row.names = FALSE,
      na = "",
      fileEncoding = "UTF-8"
    )
  }
}



# ==============================================================================
# 3. Read raw CLHLS data
# ==============================================================================

raw <- utils::read.csv(
  csv_path,
  fileEncoding = "GB18030",
  na.strings = c(
    "",
    "NA"
  ),
  strip.white = TRUE,
  stringsAsFactors = FALSE,
  check.names = FALSE
)


cat(
  "原始数据：",
  nrow(raw),
  "行，",
  ncol(raw),
  "列。\n"
)


if (nrow(raw) != 15874L) {

  warning(
    "原始样本量不是预期的15874。当前N=",
    nrow(raw)
  )
}


if (anyDuplicated(raw$id) > 0) {

  stop(
    "发现重复ID。重复ID数量：",
    sum(
      duplicated(raw$id)
    )
  )
}



# ==============================================================================
# 4. Check raw variables required by this script
# ==============================================================================

required_raw <- unique(
  c(

    # metadata
    "id",
    "yearin",
    "monthin",
    "dayin",
    "trueage",

    # demographic
    "a1",
    "a2",

    # psychological / health
    "b11",
    "b12",
    "b21",
    "b22",
    "b23",
    "b26",
    "b310a",

    # GAD-7
    paste0(
      "b4",
      1:7
    ),

    # CMMSE
    "c11",
    "c12",
    "c13",
    "c14",
    "c15",
    "c16",

    "c21a",
    "c21b",
    "c21c",

    "c31a",
    "c31b",
    "c31c",
    "c31d",
    "c31e",

    "c32",

    "c41a",
    "c41b",
    "c41c",

    "c51a",
    "c51b",
    "c52",
    "c53a",
    "c53b",
    "c53c",

    # CESD
    paste0(
      "b3",
      1:9
    ),

    # dietary
    "d1",
    "d31",
    "d32",
    "d4meat2",
    "d4fish2",
    "d4egg2",
    "d4bean2",
    "d4veg2",
    "d4suga2",
    "d4garl2",
    "d4milk1",
    "d4nut1",
    "d4alga1",
    "d4tea2",

    # ventilation
    "a5411",
    "a5412",
    "a5413",
    "a5414",

    # lifestyle
    "d71",
    "d81",
    "d101",
    "d11d",
    "d11h",

    # ADL
    paste0(
      "e",
      1:6
    ),

    # socioeconomic
    "residenc",
    "hukou",
    "f1",
    "yb32",
    "f41",
    "f64a",
    "f64e",
    "f35",
    "f10",
    "f652b",

    # sensory / fall / toothache
    "g1",
    "g106",
    "g4c1",
    "g24",

    # chronic diseases
    "g15a1",
    "g15b1",
    "g15c1",
    "g15d1",
    "g15e1",
    "g15f1",
    "g15g1",
    "g15h1",
    "g15i1",
    "g15j1",
    "g15k1",
    "g15l1",
    "g15m1",
    "g15n1",
    "g15p1",
    "g15q1",
    "g15r1",
    "g15s1",
    "g15t1",
    "g15w1",
    "g15x1"
  )
)


missing_columns <- setdiff(
  required_raw,
  names(raw)
)


if (length(missing_columns) > 0) {

  stop(
    "原始CSV缺少以下必需变量：\n",
    paste(
      missing_columns,
      collapse = ", "
    )
  )
}



# ==============================================================================
# 5. Metadata
# ==============================================================================

meta <- data.frame(

  id =
    as.character(
      raw$id
    ),

  yearin =
    to_num(
      raw$yearin
    ),

  monthin =
    to_num(
      raw$monthin
    ),

  dayin =
    to_num(
      raw$dayin
    ),

  age_continuous =
    to_num(
      raw$trueage
    ),

  stringsAsFactors = FALSE
)



# ==============================================================================
# 6. GAD-7
# ==============================================================================

gad_items <- as.data.frame(

  lapply(

    raw[
      paste0(
        "b4",
        1:7
      )
    ],

    function(x) {

      recode_num(
        x,
        c(
          "0" = 0,
          "1" = 1,
          "2" = 2,
          "3" = 3
        )
      )
    }
  )
)


names(gad_items) <- paste0(
  "gad_item",
  1:7
)


gad7_total <- row_sum_complete(
  gad_items
)



# ==============================================================================
# 7. CMMSE
# ==============================================================================

cmmse_binary_vars <- c(

  "c11",
  "c12",
  "c13",
  "c14",
  "c15",

  "c21a",
  "c21b",
  "c21c",

  "c31a",
  "c31b",
  "c31c",
  "c31d",
  "c31e",

  "c41a",
  "c41b",
  "c41c",

  "c51a",
  "c51b",
  "c52",
  "c53a",
  "c53b",
  "c53c"
)


cmmse_binary <- as.data.frame(

  lapply(

    raw[
      cmmse_binary_vars
    ],

    function(x) {

      recode_num(
        x,
        c(
          "0" = 0,
          "1" = 1,
          "8" = 0
        )
      )
    }
  )
)



# ------------------------------------------------------------------------------
# c32：figure copying
#
# 0 wrong
# 1 correct
# 8 cannot use pen
# 9 unable because disability
#
# Retain the original analysis handling:
# Codes 8 and 9 are both scored as 0
# ------------------------------------------------------------------------------

cmmse_c32 <- recode_num(
  raw$c32,
  c(
    "0" = 0,
    "1" = 1,
    "8" = 0,
    "9" = 0
  )
)



# ------------------------------------------------------------------------------
# c16：
# Name foods in one minute, maximum score = 7
#
# 88 don't know -> 0
# 99 missing -> NA
# ------------------------------------------------------------------------------

c16_raw <- to_num(
  raw$c16
)


cmmse_c16 <- ifelse(

  is.na(c16_raw) |
    c16_raw == 99,

  NA_real_,

  ifelse(

    c16_raw == 88,

    0,

    pmin(
      pmax(
        c16_raw,
        0
      ),
      7
    )
  )
)


cmmse_items <- cbind(

  cmmse_binary,

  c32 =
    cmmse_c32,

  c16 =
    cmmse_c16
)


cmmse_total <- row_sum_complete(
  cmmse_items
)


cognitive_impairment <- ifelse(

  is.na(
    cmmse_total
  ),

  NA_real_,

  ifelse(
    cmmse_total < 24,
    1,
    0
  )
)



# ==============================================================================
# 8. Create candidate-predictor data frame
# ==============================================================================

p <- data.frame(
  row.names = seq_len(
    nrow(raw)
  )
)



# ==============================================================================
# 8.1 Demographics, health, and lifestyle
# 1-12
# ==============================================================================

age <- to_num(
  raw$trueage
)



# ------------------------------------------------------------------------------
# Age groups
#
# 1 = 60-70
# 2 = 71-80
# 3 = 81-90
# 4 = >90
#
# Because the final sample requires age >=65, the first group is effectively 65-70.
# ------------------------------------------------------------------------------

p$age_group <- ifelse(

  age >= 60 &
    age <= 70,

  1,

  ifelse(

    age >= 71 &
      age <= 80,

    2,

    ifelse(

      age >= 81 &
        age <= 90,

      3,

      ifelse(
        age > 90,
        4,
        NA_real_
      )
    )
  )
)



# Sex
p$gender <- recode_num(
  raw$a1,
  c(
    "1" = 1,
    "2" = 2
  )
)



# Ethnicity
# 1 = Han
# 2 = minority

a2_num <- to_num(
  raw$a2
)


p$nation <- ifelse(

  a2_num == 1,

  1,

  ifelse(
    a2_num %in% 2:8,
    2,
    NA_real_
  )
)



# Self-reported health
# very good / good = 1
# so-so / bad / very bad = 0

p$good_health <- recode_num(
  raw$b12,
  c(
    "1" = 1,
    "2" = 1,
    "3" = 0,
    "4" = 0,
    "5" = 0
  )
)



# Sleep quality
# very good = 3
# good = 2
# so-so / bad = 1
# very bad = 0

p$sleep_quality <- recode_num(
  raw$b310a,
  c(
    "1" = 3,
    "2" = 2,
    "3" = 1,
    "4" = 1,
    "5" = 0
  )
)



# Staple food type
#
# 1 = rice / wheat / rice+wheat
# 2 = corn
# 3 = other

p$staple_food <- recode_num(
  raw$d1,
  c(
    "1" = 1,
    "2" = 2,
    "3" = 1,
    "4" = 1,
    "5" = 3
  )
)



# Visual impairment
p$visual_impairment <- recode_num(
  raw$g1,
  c(
    "1" = 0,
    "2" = 1,
    "3" = 1,
    "4" = 1
  )
)



# Hearing impairment
p$hearing_impairment <- yes_no(
  raw$g106
)



# Physical labor
p$physical_labor <- yes_no(
  raw$d101
)



# Regular physical examination
p$regular_physical_examination <- yes_no(
  raw$f652b
)



# Current smoking
p$smoke <- yes_no(
  raw$d71
)



# Current alcohol consumption
p$drink_alcohol <- yes_no(
  raw$d81
)



# ==============================================================================
# 8.2 13 dietary variables
# 13-25
# ==============================================================================

p$fruit <- food_fruit_veg(
  raw$d31
)


p$vegetable <- food_fruit_veg(
  raw$d32
)


p$meat <- food_other(
  raw$d4meat2
)


p$fish <- food_other(
  raw$d4fish2
)


p$eggs <- food_other(
  raw$d4egg2
)


p$bean_products <- food_other(
  raw$d4bean2
)


p$salted_vegetables <- food_other(
  raw$d4veg2
)


p$sugar <- food_other(
  raw$d4suga2
)


p$garlic <- food_other(
  raw$d4garl2
)


p$milk_products <- food_other(
  raw$d4milk1
)


p$nut_products <- food_other(
  raw$d4nut1
)


p$mushrooms_or_algae <- food_other(
  raw$d4alga1
)


p$tea <- food_other(
  raw$d4tea2
)



# ==============================================================================
# 8.3 Seasonal indoor ventilation
# 26-29
#
# Derived variable ventilation_frequency has been removed
# ==============================================================================

vent_map <- c(
  "1" = 0,
  "2" = 1,
  "3" = 1,
  "4" = 2
)


p$indoor_ventilation_spring <- recode_num(
  raw$a5411,
  vent_map
)


p$indoor_ventilation_summer <- recode_num(
  raw$a5412,
  vent_map
)


p$indoor_ventilation_autumn <- recode_num(
  raw$a5413,
  vent_map
)


p$indoor_ventilation_winter <- recode_num(
  raw$a5414,
  vent_map
)



# ==============================================================================
# 8.4 Chronic diseases
# 30-50
#
# 21 variables in total
#
# Removed:
# dementia
# breast_hyperplasia
# uterine_tumor
# chronic_disease_comorbidity
# ==============================================================================

disease_raw <- c(

  hypertension =
    "g15a1",

  diabetes =
    "g15b1",

  heart_disease =
    "g15c1",

  stroke_or_cvd =
    "g15d1",

  chronic_lung_disease =
    "g15e1",

  pulmonary_tuberculosis =
    "g15f1",

  cataract =
    "g15g1",

  glaucoma =
    "g15h1",

  cancer =
    "g15i1",

  prostate_tumor =
    "g15j1",

  gastric_or_duodenal_ulcer =
    "g15k1",

  parkinsons_disease =
    "g15l1",

  bedsore =
    "g15m1",

  arthritis =
    "g15n1",

  epilepsy =
    "g15p1",

  cholecystitis_cholelithiasis =
    "g15q1",

  dyslipidemia =
    "g15r1",

  rheumatism =
    "g15s1",

  chronic_nephritis =
    "g15t1",

  prostatic_hyperplasia =
    "g15w1",

  hepatitis =
    "g15x1"
)



# ------------------------------------------------------------------------------
# Generate 21 disease variables in batch
#
# Use single-bracket indexing deliberately,
# to avoid R syntax errors caused by double brackets being split across lines when code is copied.
# ------------------------------------------------------------------------------

for (new_name in names(disease_raw)) {

  source_var <- unname(
    disease_raw[new_name]
  )

  p[, new_name] <- yes_no(
    raw[, source_var]
  )
}



# Automatically check disease variables

missing_disease_vars <- setdiff(
  names(disease_raw),
  names(p)
)


if (length(missing_disease_vars) > 0) {

  stop(
    "以下慢病变量没有成功生成：",
    paste(
      missing_disease_vars,
      collapse = ", "
    )
  )
}


cat(
  "慢病变量成功生成：",
  length(disease_raw),
  "/21\n"
)



# ==============================================================================
# 8.5 Falls and toothache
# 51-52
# ==============================================================================

p$fall <- yes_no(
  raw$g4c1
)


p$toothache <- yes_no(
  raw$g24
)



# ==============================================================================
# 8.6 Psychological variables
# 53-56
# ==============================================================================

p$energetically <- freq_0_3(
  raw$b23
)


p$optimistic <- freq_0_3(
  raw$b21
)


p$neatly <- freq_0_3(
  raw$b22
)


p$working_condition <- freq_0_3(
  raw$b26
)


# Note:
#
# To preserve traceability to the original manuscript data columns,
# temporarily retain the previous column names:
#
# optimistic
# neatly
# working_condition
#
# However, the actual meaning of the source item should be described accurately in the final Methods section.



# ==============================================================================
# 8.7 Quality of life, reading, social activity, and marital status
# 57-60
# ==============================================================================

p$life_satisfaction <- recode_num(
  raw$b11,
  c(
    "1" = 1,
    "2" = 1,
    "3" = 0,
    "4" = 0,
    "5" = 0
  )
)


# Note:
# The official meaning of b11 is self-reported quality of life.
# The previous variable name life_satisfaction is temporarily retained here
# for traceability to the original manuscript code.



p$read <- recode_num(
  raw$d11d,
  c(
    "1" = 1,
    "2" = 1,
    "3" = 0,
    "4" = 0,
    "5" = 0
  )
)



p$socialize <- recode_num(
  raw$d11h,
  c(
    "1" = 1,
    "2" = 1,
    "3" = 0,
    "4" = 0,
    "5" = 0
  )
)



# 1 = currently married and living with spouse
# 0 = otherwise

p$marital_status <- recode_num(
  raw$f41,
  c(
    "1" = 1,
    "2" = 0,
    "3" = 0,
    "4" = 0,
    "5" = 0
  )
)



# ==============================================================================
# 8.8 CESD-10 / depression risk
# Predictor 61
# ==============================================================================

cesd_items <- data.frame(

  item1 =
    cesd_negative_item(
      raw$b31
    ),

  item2 =
    cesd_negative_item(
      raw$b32
    ),

  item3 =
    cesd_negative_item(
      raw$b33
    ),

  item4 =
    cesd_negative_item(
      raw$b34
    ),

  item5 =
    cesd_positive_item(
      raw$b35
    ),

  item6 =
    cesd_negative_item(
      raw$b36
    ),

  item7 =
    cesd_positive_item(
      raw$b37
    ),

  item8 =
    cesd_negative_item(
      raw$b38
    ),

  item9 =
    cesd_negative_item(
      raw$b39
    ),

  item10 =
    recode_num(
      raw$b310a,
      c(
        "1" = 0,
        "2" = 1,
        "3" = 2,
        "4" = 2,
        "5" = 3
      )
    )
)


cesd10_total <- row_sum_complete(
  cesd_items
)


p$depression_risk <- ifelse(

  is.na(
    cesd10_total
  ),

  NA_real_,

  ifelse(
    cesd10_total >= 10,
    1,
    0
  )
)



# ==============================================================================
# 8.9 Katz ADL
# Predictor 62
# ==============================================================================

adl_items <- as.data.frame(

  lapply(

    raw[
      paste0(
        "e",
        1:6
      )
    ],

    function(x) {

      recode_num(
        x,
        c(
          "1" = 0,
          "2" = 1,
          "3" = 1
        )
      )
    }
  )
)


names(adl_items) <- c(
  "bathing",
  "dressing",
  "toileting",
  "indoor_transfer",
  "continence",
  "eating"
)


p$disability_risk <- row_any_positive(
  adl_items
)



# ==============================================================================
# 8.10 Socioeconomic variables
# 63-70
# ==============================================================================


# ------------------------------------------------------------------------------
# Residential area
#
# 1 city
# 2 town
# 3 rural
# ------------------------------------------------------------------------------

p$residential_area <- recode_num(
  raw$residenc,
  c(
    "1" = 1,
    "2" = 2,
    "3" = 3
  )
)



# ------------------------------------------------------------------------------
# Hukou
# ------------------------------------------------------------------------------

p$hukou <- recode_num(
  raw$hukou,
  c(
    "1" = 1,
    "2" = 2
  )
)



# ------------------------------------------------------------------------------
# Education
#
# 0 = 0 years
# 1 = 1-6 years
# 2 = >6 years
#
# 88 don't know
# 99 missing
# ------------------------------------------------------------------------------

education_years <- to_num(
  raw$f1
)


education_years[
  education_years %in%
    c(
      88,
      99
    )
] <- NA_real_


education_years[
  education_years < 0
] <- NA_real_


p$years_of_education <- ifelse(

  is.na(
    education_years
  ),

  NA_real_,

  ifelse(

    education_years == 0,

    0,

    ifelse(

      education_years <= 6,

      1,

      2
    )
  )
)



# ------------------------------------------------------------------------------
# Residence duration
#
# 0 = <=30 years
# 1 = >30 years
#
# 888 don't know
# 999 missing
# ------------------------------------------------------------------------------

residence_years <- to_num(
  raw$yb32
)


residence_years[
  residence_years %in%
    c(
      88,
      99,
      888,
      999
    )
] <- NA_real_


residence_years[
  residence_years < 0
] <- NA_real_


p$residence_duration <- ifelse(

  is.na(
    residence_years
  ),

  NA_real_,

  ifelse(
    residence_years <= 30,
    0,
    1
  )
)



# ------------------------------------------------------------------------------
# Old-age pension
#
# f64a:
# 0 no
# 1 yes
# ------------------------------------------------------------------------------

p$old_age_pension <- recode_num(
  raw$f64a,
  c(
    "0" = 0,
    "1" = 1
  )
)



# ------------------------------------------------------------------------------
# Annual household income
#
# f35:
# 88888 = don't know
# 99998 = more than 100000 RMB
# 99999 = missing
#
# Grouping:
# 1 = <30000
# 2 = 30000-70000
# 3 = >70000
#
# 99998 is treated as a valid high-income response.
# ------------------------------------------------------------------------------

income <- to_num(
  raw$f35
)


income[
  income %in%
    c(
      88888,
      99999
    )
] <- NA_real_


income[
  income < 0
] <- NA_real_


p$annual_household_income <- ifelse(

  is.na(
    income
  ),

  NA_real_,

  ifelse(

    income < 30000,

    1,

    ifelse(

      income <= 70000,

      2,

      3
    )
  )
)



# ------------------------------------------------------------------------------
# Medical insurance
#
# Continue using f64e to match the original manuscript variable.
#
# Note:
# f64e specifically corresponds to urban employee/resident medical insurance,
# and does not represent all types of medical insurance.
# ------------------------------------------------------------------------------

p$medical_insurance <- recode_num(
  raw$f64e,
  c(
    "0" = 0,
    "1" = 1
  )
)



# ------------------------------------------------------------------------------
# Number of children
#
# 0 = none
# 1 = 1-2
# 2 = >=3
#
# 88 don't know
# 99 missing
# ------------------------------------------------------------------------------

children <- to_num(
  raw$f10
)


children[
  children %in%
    c(
      88,
      99
    )
] <- NA_real_


children[
  children < 0
] <- NA_real_


p$number_of_children <- ifelse(

  is.na(
    children
  ),

  NA_real_,

  ifelse(

    children == 0,

    0,

    ifelse(

      children <= 2,

      1,

      2
    )
  )
)



# ==============================================================================
# 9. Fix the 70 candidate predictors and their order
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



# ------------------------------------------------------------------------------
# Check the number of predictors
# ------------------------------------------------------------------------------

if (length(predictor_names) != 70L) {

  stop(
    "候选预测变量数量不是70！当前数量=",
    length(predictor_names)
  )
}



# ------------------------------------------------------------------------------
# Check whether all variables were generated successfully
# ------------------------------------------------------------------------------

missing_predictors <- setdiff(
  predictor_names,
  names(p)
)


if (length(missing_predictors) > 0) {

  stop(
    "以下候选变量没有成功生成：\n",
    paste(
      missing_predictors,
      collapse = ", "
    )
  )
}


p <- p[
  predictor_names
]


cat(
  "70项候选预测变量全部生成成功。\n"
)



# ==============================================================================
# 10. Record missingness among the 70 predictors
#
# IMPORTANT:
# Record only; do not exclude participants.
# ==============================================================================

predictor_missing_n <- rowSums(
  is.na(p)
)


predictor_missing_rate <-
  predictor_missing_n /
  length(
    predictor_names
  )



# ==============================================================================
# 11. Final sample selection
# ==============================================================================


# ------------------------------------------------------------------------------
# Missing age
# ------------------------------------------------------------------------------

exclude_age_missing <- is.na(
  meta$age_continuous
)



# ------------------------------------------------------------------------------
# Age <65
# ------------------------------------------------------------------------------

exclude_age_lt65 <-

  !is.na(
    meta$age_continuous
  ) &

  meta$age_continuous < 65



# ------------------------------------------------------------------------------
# Age >=65
# ------------------------------------------------------------------------------

keep_after_age <-

  !is.na(
    meta$age_continuous
  ) &

  meta$age_continuous >= 65



# ------------------------------------------------------------------------------
# Missing GAD-7
# ------------------------------------------------------------------------------

exclude_gad_missing <-

  keep_after_age &

  is.na(
    gad7_total
  )



# ------------------------------------------------------------------------------
# GAD-7 <4
# ------------------------------------------------------------------------------

exclude_gad_lt4 <-

  keep_after_age &

  !is.na(
    gad7_total
  ) &

  gad7_total < 4



# ------------------------------------------------------------------------------
# Age >=65 + GAD-7 >=4
# ------------------------------------------------------------------------------

keep_after_gad <-

  keep_after_age &

  !is.na(
    gad7_total
  ) &

  gad7_total >= 4



# ------------------------------------------------------------------------------
# CMMSE cannot be fully scored
# ------------------------------------------------------------------------------

exclude_cmmse_missing <-

  keep_after_gad &

  is.na(
    cmmse_total
  )



# ------------------------------------------------------------------------------
# Final analysis sample
# ------------------------------------------------------------------------------

analysis_flag <-

  keep_after_gad &

  !is.na(
    cmmse_total
  )



# ==============================================================================
# 12. Combine complete analysis data
# ==============================================================================

analysis_data <- cbind(

  meta,

  gad_items,

  gad7_total =
    gad7_total,

  cesd10_total =
    cesd10_total,

  cmmse_total =
    cmmse_total,

  cognitive_impairment =
    cognitive_impairment,

  p,

  predictor_missing_n =
    predictor_missing_n,

  predictor_missing_rate =
    predictor_missing_rate,

  exclude_age_missing =
    as.integer(
      exclude_age_missing
    ),

  exclude_age_lt65 =
    as.integer(
      exclude_age_lt65
    ),

  exclude_gad_missing =
    as.integer(
      exclude_gad_missing
    ),

  exclude_gad_lt4 =
    as.integer(
      exclude_gad_lt4
    ),

  exclude_cmmse_missing =
    as.integer(
      exclude_cmmse_missing
    ),

  analysis_flag =
    as.integer(
      analysis_flag
    )
)



# ==============================================================================
# 13. Final analysis sample
# ==============================================================================

final_data <- analysis_data[
  analysis_flag,
  ,
  drop = FALSE
]



# ==============================================================================
# 14. Record interview-year grouping only
#
# Note:
# Do not label 2019 directly as temporal validation here.
# The machine-learning validation strategy is determined separately later.
# ==============================================================================

final_data$interview_year_group <- ifelse(

  final_data$yearin %in%
    c(
      2017,
      2018
    ),

  "2017_2018",

  ifelse(
    final_data$yearin == 2019,
    "2019",
    "other"
  )
)



# ==============================================================================
# 15. Sample-selection counts
# ==============================================================================

n0 <- nrow(raw)


n_after_age_available <-
  n0 -
  sum(
    exclude_age_missing
  )


n_after_age <-
  sum(
    keep_after_age
  )


n_after_gad_available <-
  n_after_age -
  sum(
    exclude_gad_missing
  )


n_after_gad <-
  sum(
    keep_after_gad
  )


n_final <-
  sum(
    analysis_flag
  )



# ==============================================================================
# 16. Sample-selection log
# ==============================================================================

flow <- data.frame(

  step = c(

    "Original CLHLS sample",

    "Exclude missing validated age",

    "Exclude age <65 years",

    "Exclude incomplete GAD-7",

    "Exclude GAD-7 <4",

    "Age >=65 and GAD-7 >=4",

    "Exclude unavailable CMMSE",

    "Final analysis sample"
  ),

  excluded_n = c(

    0,

    sum(
      exclude_age_missing
    ),

    sum(
      exclude_age_lt65
    ),

    sum(
      exclude_gad_missing
    ),

    sum(
      exclude_gad_lt4
    ),

    0,

    sum(
      exclude_cmmse_missing
    ),

    0
  ),

  remaining_n = c(

    n0,

    n_after_age_available,

    n_after_age,

    n_after_gad_available,

    n_after_gad,

    n_after_gad,

    n_final,

    n_final
  ),

  stringsAsFactors = FALSE
)



# ==============================================================================
# 17. Final sample interview-year distribution
# ==============================================================================

year_distribution <- as.data.frame(

  table(
    final_data$yearin,
    useNA = "ifany"
  )
)


names(
  year_distribution
) <- c(
  "interview_year",
  "n"
)



# ==============================================================================
# 18. Quality-control table for the 70 predictors
# ==============================================================================

qc <- data.frame(

  variable_name =
    predictor_names,

  nonmissing_n =
    vapply(

      final_data[
        predictor_names
      ],

      function(x) {

        sum(
          !is.na(x)
        )
      },

      numeric(1)
    ),

  missing_n =
    vapply(

      final_data[
        predictor_names
      ],

      function(x) {

        sum(
          is.na(x)
        )
      },

      numeric(1)
    ),

  missing_percent =
    round(

      100 *

        vapply(

          final_data[
            predictor_names
          ],

          function(x) {

            mean(
              is.na(x)
            )
          },

          numeric(1)
        ),

      2
    ),

  min =
    vapply(

      final_data[
        predictor_names
      ],

      function(x) {

        if (
          all(
            is.na(x)
          )
        ) {

          NA_real_

        } else {

          min(
            x,
            na.rm = TRUE
          )
        }
      },

      numeric(1)
    ),

  max =
    vapply(

      final_data[
        predictor_names
      ],

      function(x) {

        if (
          all(
            is.na(x)
          )
        ) {

          NA_real_

        } else {

          max(
            x,
            na.rm = TRUE
          )
        }
      },

      numeric(1)
    ),

  unique_nonmissing =
    vapply(

      final_data[
        predictor_names
      ],

      function(x) {

        length(
          unique(
            x[
              !is.na(x)
            ]
          )
        )
      },

      numeric(1)
    ),

  stringsAsFactors = FALSE
)


qc <- qc[
  order(
    -qc$missing_percent
  ),
  ,
  drop = FALSE
]



# ==============================================================================
# 19. Variable dictionary for the 70 candidate predictors
# ==============================================================================

raw_source <- c(

  # 1-12
  "trueage",
  "a1",
  "a2",
  "b12",
  "b310a",
  "d1",
  "g1",
  "g106",
  "d101",
  "f652b",
  "d71",
  "d81",

  # 13-25
  "d31",
  "d32",
  "d4meat2",
  "d4fish2",
  "d4egg2",
  "d4bean2",
  "d4veg2",
  "d4suga2",
  "d4garl2",
  "d4milk1",
  "d4nut1",
  "d4alga1",
  "d4tea2",

  # 26-29
  "a5411",
  "a5412",
  "a5413",
  "a5414",

  # 30-50
  "g15a1",
  "g15b1",
  "g15c1",
  "g15d1",
  "g15e1",
  "g15f1",
  "g15g1",
  "g15h1",
  "g15i1",
  "g15j1",
  "g15k1",
  "g15l1",
  "g15m1",
  "g15n1",
  "g15p1",
  "g15q1",
  "g15r1",
  "g15s1",
  "g15t1",
  "g15w1",
  "g15x1",

  # 51-52
  "g4c1",
  "g24",

  # 53-56
  "b23",
  "b21",
  "b22",
  "b26",

  # 57-60
  "b11",
  "d11d",
  "d11h",
  "f41",

  # 61-62
  "b31-b39+b310a",
  "e1-e6",

  # 63-70
  "residenc",
  "hukou",
  "f1",
  "yb32",
  "f64a",
  "f35",
  "f64e",
  "f10"
)


if (length(raw_source) != 70L) {

  stop(
    "字段字典不是70项！当前数量=",
    length(raw_source)
  )
}


dictionary <- data.frame(

  order =
    1:70,

  variable_name =
    predictor_names,

  raw_source =
    raw_source,

  stringsAsFactors = FALSE
)



# ==============================================================================
# 20. Export all results
# ==============================================================================

write_csv_blank(
  analysis_data,
  out_all
)


write_csv_blank(
  final_data,
  out_final
)


write_csv_blank(
  flow,
  out_flow
)


write_csv_blank(
  qc,
  out_qc
)


write_csv_blank(
  year_distribution,
  out_year
)


write_csv_blank(
  dictionary,
  out_dict
)



# ==============================================================================
# 21. Automatically print results
# ==============================================================================

cat(
  "\n============================================\n"
)

cat(
  "CLHLS 70项候选变量数据整理完成\n"
)

cat(
  "============================================\n\n"
)


cat(
  "候选预测变量：",
  length(
    predictor_names
  ),
  "项\n"
)


cat(
  "原始样本：",
  n0,
  "人\n"
)


cat(
  "排除年龄缺失：",
  sum(
    exclude_age_missing
  ),
  "人\n"
)


cat(
  "排除年龄<65岁：",
  sum(
    exclude_age_lt65
  ),
  "人\n"
)


cat(
  "年龄>=65岁后：",
  n_after_age,
  "人\n"
)


cat(
  "排除GAD-7缺失：",
  sum(
    exclude_gad_missing
  ),
  "人\n"
)


cat(
  "排除GAD-7<4：",
  sum(
    exclude_gad_lt4
  ),
  "人\n"
)


cat(
  "GAD-7>=4后：",
  n_after_gad,
  "人\n"
)


cat(
  "排除CMMSE无法计分：",
  sum(
    exclude_cmmse_missing
  ),
  "人\n"
)


cat(
  "最终分析样本：",
  n_final,
  "人\n\n"
)


cat(
  "最终样本年份分布：\n"
)

print(
  year_distribution
)


cat(
  "\n缺失率最高的15个候选变量：\n"
)

print(
  head(
    qc,
    15
  )
)



# ==============================================================================
# 22. Automatically compare results with verified counts from the current raw data
# ==============================================================================

expected_n0 <- 15874L
expected_age <- 15771L
expected_gad <- 2242L
expected_final <- 2081L


if (n0 != expected_n0) {

  warning(
    "原始样本量异常：实际=",
    n0,
    "；预期=",
    expected_n0
  )
}


if (n_after_age != expected_age) {

  warning(
    "年龄筛选后人数异常：实际=",
    n_after_age,
    "；预期=",
    expected_age
  )
}


if (n_after_gad != expected_gad) {

  warning(
    "GAD-7筛选后人数异常：实际=",
    n_after_gad,
    "；预期=",
    expected_gad
  )
}


if (n_final != expected_final) {

  warning(
    "最终样本量异常：实际=",
    n_final,
    "；预期=",
    expected_final,
    "。请检查数据版本或编码，不要修改筛选标准凑人数。"
  )
}



# ------------------------------------------------------------------------------
# Check interview-year counts
# ------------------------------------------------------------------------------

expected_year <- c(
  "2017" = 158,
  "2018" = 1554,
  "2019" = 369
)


actual_year <- table(
  final_data$yearin
)


for (yr in names(expected_year)) {

  if (yr %in% names(actual_year)) {

    actual_n <- as.integer(
      actual_year[yr]
    )

  } else {

    actual_n <- 0L
  }


  if (
    actual_n !=
      expected_year[yr]
  ) {

    warning(
      "年份",
      yr,
      "人数异常：实际=",
      actual_n,
      "；预期=",
      expected_year[yr]
    )
  }
}



# ==============================================================================
# 23. Final result summary
# ==============================================================================

cat(
  "\n============================================\n"
)


cat(
  "预期最终结果：\n"
)


cat(
  "原始样本：15874\n"
)


cat(
  "年龄>=65岁：15771\n"
)


cat(
  "GAD-7>=4：2242\n"
)


cat(
  "最终CMMSE可计分样本：2081\n"
)


cat(
  "2017：158\n"
)


cat(
  "2018：1554\n"
)


cat(
  "2019：369\n"
)


cat(
  "候选预测变量：70项\n"
)


cat(
  "============================================\n"
)


cat(
  "\n输出目录：\n",
  data_dir,
  "\n"
)


cat(
  "\n最终分析文件：\n",
  out_final,
  "\n"
)


cat(
  "\n质控文件：\n",
  out_qc,
  "\n"
)


cat(
  "\n此阶段未对70项预测变量进行：",
  "缺失插补、标准化、SMOTE、P值筛选或LASSO。\n",
  sep = ""
)


cat(
  "后续机器学习预处理应在训练/CV流程内部完成。\n"
)
