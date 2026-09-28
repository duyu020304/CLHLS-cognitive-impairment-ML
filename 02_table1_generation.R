# ==============================================================================
# Participant characteristics (Table 1)
# Public-sharing version: Chinese comments translated to English.
# Analysis logic is unchanged; only comments and the local absolute project path were edited.
# Set data_dir to the appropriate local project directory before running.
# ==============================================================================

# ============================================================
# Table 1
# Participant characteristics according to cognitive impairment
#
# Variables:
# Age (continuous)
# Sex
# Education
# Marital status
# ADL disability
# Hearing impairment
# Visual impairment
#
# ============================================================


# ============================================================
# 1. Packages
# ============================================================

packages <- c(
  "readr",
  "dplyr",
  "gtsummary",
  "flextable",
  "officer"
)

missing_packages <- packages[
  !vapply(
    packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if(length(missing_packages) > 0){
  
  install.packages(
    missing_packages,
    dependencies = TRUE
  )
  
}


library(readr)
library(dplyr)
library(gtsummary)
library(flextable)
library(officer)



# ============================================================
# 2. File path
# ============================================================

data_dir <- "path/to/project"


# Automatically locate the final analysis dataset

data_files <- list.files(
  data_dir,
  pattern = "最终分析样本.*\\.csv$",
  full.names = TRUE
)


if(length(data_files) == 0){
  
  stop(
    "没有找到最终分析样本CSV文件，请检查路径"
  )
  
}


if(length(data_files) > 1){
  
  print(data_files)
  
  stop(
    "发现多个最终分析样本文件，请确认"
  )
  
}


data_path <- data_files[1]


output_word <- file.path(
  data_dir,
  "Table1_Participant_characteristics.docx"
)


output_csv <- file.path(
  data_dir,
  "Table1_Participant_characteristics.csv"
)



cat(
  "读取数据：",
  data_path,
  "\n"
)



# ============================================================
# 3. Read data
# ============================================================


dat <- read_csv(
  data_path,
  show_col_types = FALSE
)


cat(
  "样本量:",
  nrow(dat),
  "\n"
)



# ============================================================
# 4. Variable check
# ============================================================


required_vars <- c(
  
  "cognitive_impairment",
  
  "age_continuous",
  
  "gender",
  
  "years_of_education",
  
  "marital_status",
  
  "disability_risk",
  
  "hearing_impairment",
  
  "visual_impairment"
  
)


missing_vars <- setdiff(
  required_vars,
  names(dat)
)


if(length(missing_vars)>0){
  
  stop(
    paste(
      "缺少变量:",
      paste(
        missing_vars,
        collapse=", "
      )
    )
  )
  
}




# ============================================================
# 5. Recode variables
# ============================================================


dat_table1 <- dat %>%
  
  mutate(
    
    
    # ----------------------------
    # Outcome
    # ----------------------------
    
    cognitive_group = factor(
      
      cognitive_impairment,
      
      levels=c(0,1),
      
      labels=c(
        "No cognitive impairment",
        "Cognitive impairment"
      )
      
    ),
    
    
    
    # ----------------------------
    # Age
    # continuous
    # ----------------------------
    
    age_continuous =
      as.numeric(age_continuous),
    
    
    
    # ----------------------------
    # Sex
    # ----------------------------
    
    gender=factor(
      
      gender,
      
      levels=c(1,2),
      
      labels=c(
        "Male",
        "Female"
      )
      
    ),
    
    
    
    # ----------------------------
    # Education
    # ----------------------------
    
    years_of_education=factor(
      
      years_of_education,
      
      levels=c(0,1,2),
      
      labels=c(
        
        "0 years",
        
        "1–6 years",
        
        ">6 years"
        
      )
      
    ),
    
    
    
    # ----------------------------
    # Marital status
    # ----------------------------
    
    marital_status=factor(
      
      marital_status,
      
      levels=c(0,1),
      
      labels=c(
        
        "Other",
        
        "Currently married and living with spouse"
        
      )
      
    ),
    
    
    
    # ----------------------------
    # ADL disability
    # ----------------------------
    
    disability_risk=factor(
      
      disability_risk,
      
      levels=c(0,1),
      
      labels=c(
        
        "No",
        
        "Yes"
        
      )
      
    ),
    
    
    
    # ----------------------------
    # Hearing impairment
    # ----------------------------
    
    hearing_impairment=factor(
      
      hearing_impairment,
      
      levels=c(0,1),
      
      labels=c(
        
        "No",
        
        "Yes"
        
      )
      
    ),
    
    
    
    # ----------------------------
    # Visual impairment
    # ----------------------------
    
    visual_impairment=factor(
      
      visual_impairment,
      
      levels=c(0,1),
      
      labels=c(
        
        "No",
        
        "Yes"
        
      )
      
    )
    
  )





# ============================================================
# 6. Generate Table 1
# ============================================================


table1 <- dat_table1 %>%
  
  
  select(
    
    cognitive_group,
    
    age_continuous,
    
    gender,
    
    years_of_education,
    
    marital_status,
    
    disability_risk,
    
    hearing_impairment,
    
    visual_impairment
    
  ) %>%
  
  
  tbl_summary(
    
    
    by = cognitive_group,
    
    
    
    # ========================================================
    # Force variable types
    # Prevent binary variables from displaying only the Yes category
    # ========================================================
    
    type = list(
      
      age_continuous ~ "continuous",
      
      gender ~ "categorical",
      
      years_of_education ~ "categorical",
      
      marital_status ~ "categorical",
      
      disability_risk ~ "categorical",
      
      hearing_impairment ~ "categorical",
      
      visual_impairment ~ "categorical"
      
    ),
    
    
    
    statistic=list(
      
      
      age_continuous ~
        "{mean} ({sd})",
      
      
      all_categorical() ~
        "{n} ({p}%)"
      
      
    ),
    
    
    
    digits=list(
      
      age_continuous ~ 1,
      
      all_categorical() ~ 1
      
    ),
    
    
    
    missing="ifany",
    
    
    
    label=list(
      
      
      age_continuous ~
        "Age, years",
      
      
      gender ~
        "Sex",
      
      
      years_of_education ~
        "Education",
      
      
      marital_status ~
        "Marital status",
      
      
      disability_risk ~
        "ADL disability",
      
      
      hearing_impairment ~
        "Hearing impairment",
      
      
      visual_impairment ~
        "Visual impairment"
      
      
    )
    
    
  ) %>%
  
  
  
  add_overall(
    last = FALSE
  ) %>%
  
  
  
  add_p(
    
    test=list(
      
      age_continuous ~ "t.test",
      
      all_categorical() ~ "chisq.test"
      
    )
    
  ) %>%
  
  
  
  modify_header(
    
    label ~ "**Characteristic**",
    
    stat_0 ~ "**Overall**",
    
    p.value ~ "**P value**"
    
  ) %>%
  
  
  
  bold_labels()





# ============================================================
# 7. Print result
# ============================================================


table1





# ============================================================
# 8. Export CSV
# ============================================================


table1_df <- as_tibble(table1)


write_excel_csv(
  
  table1_df,
  
  output_csv
  
)



# ============================================================
# 9. Export Word
# ============================================================


table1_ft <- as_flex_table(table1)


doc <- read_docx()



doc <- body_add_par(
  
  doc,
  
  "Table 1. Participant characteristics according to cognitive impairment status",
  
  style="heading 1"
  
)



doc <- body_add_flextable(
  
  doc,
  
  table1_ft
  
)



doc <- body_add_par(
  
  doc,
  
  paste0(
    
    "Continuous variables are presented as mean (SD). ",
    
    "Categorical variables are presented as n (%). ",
    
    "P values are descriptive comparisons between groups "
    
  )
  
)



print(
  
  doc,
  
  target = output_word
  
)



cat(
  "\n完成\n"
)


cat(
  "Word:",
  output_word,
  "\n"
)


cat(
  "CSV:",
  output_csv,
  "\n"
)
