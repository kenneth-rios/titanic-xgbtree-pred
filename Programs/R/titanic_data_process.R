# Data read-in, pooling train/test data
#
# Data wrangling to get all the predictors we desire for xgboost model. Keep as data frame for now with factor variables
# in order to convert to one-hot encoding using model.matrix()
#
# Do a basic imputation for fare and embarked (see example kaggle code, use same methodology)
#
# Using the caret package, impute missing values for Age using bagged trees jointly for the training and test data
#

### Clear memory
rm(list=ls())

### Load packages
library(caret)
library(dplyr)
library(RANN)

### Define root directory
root <- "C:\\Users\\kenri\\OneDrive\\titanic_survival_pred"



# Read in data
train <- read.csv(paste0(root, "\\Raw Data\\train.csv"), stringsAsFactors = FALSE)
train$data <- "train"

test <- read.csv(paste0(root, "\\Raw Data\\test.csv"), stringsAsFactors = FALSE)
test$data <- "test"

# Pool data for data wrangling and imputation
titanic <- bind_rows(train, test)



### Parse Name variable using regex to generate a title variable
titanic$title <- gsub("^.*, (.*?)\\..*$", "\\1", titanic$Name)
# I grouped several names together according to the meaning of the honorifics and
# relative frequencies of survival. See:
#
# table(titanic$Pclass, titanic$title)
# table(titanic$Survived, titanic$title)

titanic$title <- gsub("Mlle|Ms", "Miss", titanic$title)
titanic$title <- gsub("Dona|Mme", "Lady", titanic$title)
titanic$title <- gsub("Don|Jonkheer", "Sir", titanic$title)
#titanic$title <- gsub("Col|Major", "Deputy", titanic$title)  # This grouping is debatable, but we'll see.



### Use the presence of a nickname in quotes ("...") as a predictor. 
# All married women have their names in parentheses, so the presence of parentheses for women has no meaning
# other than conveying their marital status, which is already captured by title. Later I might look at nicknames 
# ("..." or inside parentheses) only for males or for 2nd/3rd class passengers, although decision tree ensembles
# should model interactions between features very well.
titanic$nickname <- ifelse(grepl("(.*?)\"(.*?)\"(.*?)", titanic$Name), 1, 0)



### I believe sibsp and parch should be left separate and not combined to form "family_size". 
### There are different relationships that parents have with their children and what siblings have
### for each other. Combining into a single number would treat parents with many children in danger
### the same as a child with many brothers/sisters in danger, which is clearly not comparable.



### Use unique Ticket number to identify family groups (seems to works very well via inspection)
titanic <- transform(titanic, familyID = match(Ticket, unique(Ticket)))

titanic$Ticket <- NULL



### Use the first letter of Cabin and simply bin missing cabins to the average cabin level of the passenger class
titanic$cabin <- substr(titanic$Cabin, 1, 1) 

titanic$cabin <- ifelse(titanic$Pclass == 1 & 
                        (titanic$cabin == "" | titanic$Cabin == "T"), "C", titanic$cabin)
titanic$cabin <- ifelse(titanic$Pclass == 2 & 
                        titanic$cabin == "", "E", titanic$cabin)
titanic$cabin <- ifelse(titanic$Pclass == 3 & 
                        titanic$cabin == "", "F", titanic$cabin)

titanic$Cabin <- NULL



### Impute two missing embarkments by noting that those passengers purchased first class tickets for 80 units.
### Choose the embarkment with the median fare for first class tickets closest to 80, which is "C":
# titanic[titanic$Embarked == "", ]

diff <- 1E8

imputed_embark <- lapply(unique(titanic$Embarked)[unique(titanic$Embarked) != ""],

  function(x){

    if(abs(mean(titanic[titanic$Embarked == "", ]$Fare) -
           median(titanic[titanic$Pclass == 1 & titanic$Embarked == x, ]$Fare)) < diff){

      diff <<- abs(mean(titanic[titanic$Embarked == "", ]$Fare) -
                   median(titanic[titanic$Pclass == 1 & titanic$Embarked == x, ]$Fare))

      imputed_embark <<- x
    }

    return(imputed_embark)
  })

imputed_embark <- imputed_embark[[length(imputed_embark)]]
titanic[titanic$Embarked == "", ]$Embarked <- imputed_embark



### Impute single missing value for Fare using the average fare of the corresponding Pclass, Embarked, and 
### approximate age group (+/- 5 years). The ticket's price was determined primarily by the passenger class, 
### the port of embarkment, and the age of the passenger.
# titanic[is.na(titanic$Fare), ]

imputed_fare <- mean(titanic[titanic$Pclass == titanic[is.na(titanic$Fare),]$Pclass &
                               titanic$Embarked == titanic[is.na(titanic$Fare),]$Embarked &
                               (titanic$Age >= titanic[is.na(titanic$Fare),]$Age - 5 &
                                  titanic$Age <= titanic[is.na(titanic$Fare),]$Age + 5), ]$Fare, na.rm = TRUE)

titanic[is.na(titanic$Fare), ]$Fare <- imputed_fare



### Pre-process titanic data for imputation of Age variable
# Remove defunct variables
titanic$Name <- NULL
titanic$PassengerId <- NULL

# Convert categorical variables to factors
factors <- c('Pclass', 'Sex', 'Embarked', 'title', 'familyID', 'cabin')
titanic[factors] <- lapply(titanic[factors], function(x) as.factor(x))

# Store variables which should be deleted for Age imputation: I posit that, given our available data,
# Age only makes sense as a function of passenger class, sex, number of siblings/spouses, number of parents/children,
# the fare, name title, and presence of a nickname. See theory section for some justifications why.
#
# I think including Sex is debatable, but we'll see.
stored <- select(titanic, -c(Age, Pclass, SibSp, Parch, Fare, title, nickname))
titanic <- select(titanic, -c(Survived, data, Sex, Embarked, familyID, cabin))



### Impute Age using caret::predict(preProcess(method = "bagImpute"))
# Impute Age variable by fitting a bagged tree model, as a function of all other numerical predictors
#set.seed(1)

titanic <- predict(preProcess(titanic, method = c("bagImpute")), titanic)

# Pull back in stored variables
titanic <- cbind(titanic, stored)
rm(stored)

# Generate Age**2, representing diminishing returns w.r.t. the effort/emphasis placed in saving a life.
# I'm concerned how the decision tree ensemble will handle quadratic forms, but we'll see.
titanic$age_sq <- titanic$Age ** 2   



### Prepare training and test sets for one-hot encoding
# Rename variables
colnames(titanic) <- tolower(colnames(titanic))
colnames(titanic)[colnames(titanic) == "familyid"] <- "familyID"

# Split titanic data back into training and test sets
train <- titanic[titanic$data == "train", ]
train$data <- NULL

test <- titanic[titanic$data == "test", ]
test$data <- test$survived <- NULL

# Reorder variables in training and test set data frames
train <- train[, c("pclass", "title", "sex", "age", "age_sq", "fare", 
                   "sibsp", "parch", "nickname", "embarked", "cabin", 
                   "familyID", "survived")]

test <- test[, c("pclass", "title", "sex", "age", "age_sq", "fare", 
                 "sibsp", "parch", "nickname", "embarked", "cabin",
                 "familyID")]



### Export data matrices as .RData files
str(train)
str(test)

# Save as .RData files
saveRDS(train, paste0(root, "\\Data\\train.rds"))
saveRDS(test, paste0(root, "\\Data\\test.rds"))


