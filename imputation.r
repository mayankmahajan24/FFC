library(mice)
library(caret)
temp = as.data.frame(read.csv("/Users/kapoor/Downloads/cos424_hw2/FFC/background.csv"))

# nas_cols = c(1)
# for(i in 1:12945) { nas_cols[i] = sum(is.na(temp[,i])) }
# sort(nas_cols, decreasing = TRUE)

# col_vars=c(1)
# for(i in 1:10695) { col_vars[i] = var(fuller_data[,i]) }

fuller_data <- temp[,colSums(is.na(temp)) < 2000]

# Remove factor(categorical) cols
fac_cols = split(names(fuller_data),sapply(fuller_data, function(x) paste(class(x), collapse=" ")))$factor
fuller_data = fuller_data[ , !(names(fuller_data) %in% fac_cols)]

badCols <- nearZeroVar(fuller_data)
fuller_data <- fuller_data[, -badCols]

# Reduces the features from 8941 to 8360. But reduces the missing values from 798854 to 1687
subset_data <- fuller_data[,colSums(is.na(fuller_data)) < 200]

# Reduces the features from 8941 to 8346. But reduces the missing values from 798854 to 282
subset_data <- fuller_data[,colSums(is.na(fuller_data)) < 50]

# Reduces the features from 8941 to 8344. But reduces the missing values from 798854 to 253
subset_data <- fuller_data[,colSums(is.na(fuller_data)) < 5]

# Reduces the features from 8941 to 8293. But reduces the missing values from 798854 to 72
subset_data <- fuller_data[,colSums(is.na(fuller_data)) < 3]

# Reduces the features from 8941 to 8243. But reduces the missing values from 798854 to 0
subset_data <- fuller_data[,colSums(is.na(fuller_data)) < 1]