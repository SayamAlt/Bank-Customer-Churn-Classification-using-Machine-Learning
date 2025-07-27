library(ggplot2)
library(Amelia)
library(plotly)
library(dplyr)
library(reshape2)
library(GGally)
library(ggmosaic)
library(corrplot)
library(glue)
library(caret)
library(caTools)
library(class)
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)
library(C50)
library(gbm)
library(xgboost)
library(ipred)
library(adabag)
library(nnet)
library(MASS)
library(smotefamily)
library(keras)
library(tensorflow)
library(glmnet)

# keras::install_keras() # Install Tensorflow Backend

# Load the dataset
churn.df <- read.csv('churn.csv')
print(dim(churn.df))
print(head(churn.df))
print(str(churn.df))
print(summary(churn.df))

# Checking for missing data and duplicate records
any(is.na(churn.df))
any(duplicated(churn.df))

options(contrasts = c("contr.treatment", "contr.poly"))

missmap(churn.df,main='Missingness Map',y.at=0,y.labels=NULL,col=c('yellow','black'))

# EDA (Exploratory Data Analysis)

# Convert categorical variables to factors
churn.df$churn <- factor(churn.df$churn)
churn.df$country <- factor(churn.df$country)
churn.df$gender <- factor(churn.df$gender)
churn.df$credit_card <- factor(churn.df$credit_card)
churn.df$active_member <- factor(churn.df$active_member)
churn.df$tenure <- factor(churn.df$tenure)
churn.df$products_number <- factor(churn.df$products_number)

# Univariate Analysis
ggplotly(ggplot(churn.df,aes(x=credit_score)) + geom_histogram(alpha=0.6,bins=30,color='black') + theme_bw())
ggplotly(ggplot(churn.df,aes(x=country)) + geom_bar(alpha=0.8,mapping=aes(fill=country)) + labs(x='Country',fill='Country') + theme_bw())
ggplotly(ggplot(churn.df,aes(x=gender)) + geom_bar(alpha=0.8,mapping=aes(fill=gender)) + labs(x='Gender',fill='Gender') + scale_fill_manual(values=c('darkgreen','brown')) + theme_bw())
ggplotly(ggplot(churn.df,aes(x=churn)) + geom_bar(alpha=0.8,mapping=aes(fill=churn)) + labs(x='Churn',fill='Churn') + scale_fill_manual(values=c('rosybrown','darkcyan')) + theme_bw())
ggplotly(ggplot(churn.df,aes(x=estimated_salary)) + geom_histogram(alpha=0.7,bins=50,color='black') + theme_bw())
ggplotly(ggplot(churn.df,aes(x=tenure)) + geom_bar(alpha=0.8,mapping=aes(fill=tenure)) + labs(x='Tenure',fill='Tenure') + scale_fill_manual(values=c('violet','orangered','yellow','pink','royalblue','lawngreen','sienna','gold','navy','purple','red')) + theme_bw())
ggplotly(ggplot(churn.df,aes(x=products_number)) + geom_bar(alpha=0.8,mapping=aes(fill=products_number)) + labs(x='Num Products',fill='Num Products') + scale_fill_manual(values=c('blue','red','yellow','orange')) + theme_bw())
ggplotly(ggplot(churn.df,aes(x=balance)) + geom_histogram(alpha=0.7,bins=50,color='black') + theme_bw())
ggpairs(churn.df[, c("credit_score", "age", "balance", "estimated_salary", "churn")])

# Impute outliers of balance column
churn.df$balance[churn.df$balance == 0] <- NA

churn.df <- churn.df %>%
  group_by(gender) %>%
  mutate(balance = ifelse(is.na(balance),mean(balance,na.rm=TRUE),balance)) %>%
  ungroup()

ggplotly(ggplot(churn.df,aes(x=balance)) + geom_histogram(alpha=0.7,bins=50,color='black') + theme_bw())
ggplotly(ggplot(churn.df,aes(x=age)) + geom_histogram(alpha=0.8,bins=40,color='black') + theme_bw())

# Bivariate Analysis
ggplotly(ggplot(churn.df,aes(x=country,y=credit_score,color=factor(active_member))) + 
           geom_boxplot(mapping=aes(fill=factor(active_member)),alpha=0.7,outlier.shape=16,outlier.size=2) + 
           labs(x='Country',
                y='Credit Score',
                color='Active Member',
                fill='Active Member') + theme_bw())
ggplotly(ggplot(churn.df,aes(x=gender,y=credit_score,color=factor(active_member))) + 
           geom_boxplot(mapping=aes(fill=factor(active_member)),alpha=0.7,outlier.shape=16,outlier.size=2) + 
           labs(x='Gender',
                y='Credit Score',
                color='Active Member',
                fill='Active Member') + theme_bw())
ggplotly(ggplot(churn.df,aes(x=age,y=credit_score,color=churn)) + geom_point(mapping=aes(fill=churn),alpha=0.8,size=2) + theme_bw())
ggplotly(ggplot(churn.df,aes(x=age,y=credit_score)) + geom_hex() + theme_bw())
ggplotly(ggplot(churn.df,aes(x=age,y=balance)) + geom_point(alpha=0.6) + facet_wrap(~ gender) + theme_bw())
ggplotly(ggplot(churn.df,aes(x=estimated_salary,y=balance,color=credit_card)) + geom_point(mapping=aes(fill=credit_card),alpha=0.8,size=2) + theme_bw())
ggplotly(ggplot(churn.df,aes(x=country,y=balance,fill=churn)) + geom_violin(trim=FALSE,alpha=0.7) + theme_bw())
ggplotly(ggplot(churn.df,aes(x=products_number,y=balance,color=churn)) + geom_jitter(alpha = 0.5) + theme_bw())
ggplotly(ggplot(churn.df,aes(x=products_number,fill=churn)) + geom_bar(position = "fill") + theme_bw())
ggplotly(ggplot(churn.df,aes(x=tenure,fill=churn)) + geom_bar(position='fill') + theme_bw())
ggplotly(ggplot(churn.df,aes(x=credit_card,fill=churn)) + geom_bar(position='dodge') + theme_bw())
ggplotly(ggplot(churn.df,aes(x=gender,fill=churn)) + geom_bar(position='dodge') + theme_bw())
ggplotly(ggplot(churn.df,aes(x=country,fill=credit_card)) + geom_bar() + theme_bw())
ggplotly(ggplot(churn.df) + geom_mosaic(aes(weight=1,x=product(active_member),fill=churn)) + labs(x='Is Active Member?'))
ggplotly(ggplot(churn.df, aes(x=churn,y=credit_score,fill=churn)) + geom_boxplot() + geom_jitter(width=0.2, alpha=0.4) + theme_bw())
ggplotly(ggplot(churn.df, aes(x=estimated_salary,fill=churn)) + geom_density(alpha=0.6) + theme_bw())

# Multivariate Analysis
# Correlation heatmap
cor_data <- cor(x=churn.df[,c('credit_score','age','balance','estimated_salary')], use='complete.obs')
print(cor_data)
ggplotly(ggplot(melt(cor_data), aes(Var1, Var2, fill = value)) + geom_tile())
ggplotly(ggparcoord(churn.df[, c("credit_score","age","balance","estimated_salary")]))

# Feature Engineering
churn.df$credit.age.ratio <- churn.df$credit_score / churn.df$age
churn.df$active.products <- as.numeric(churn.df$products_number) * as.numeric(as.character(churn.df$active_member)) 

# One hot encoding
one_hot_enc_cols = c('country','gender','credit_card','active_member')

f <- as.formula(paste("~", paste(one_hot_enc_cols, collapse=" + ")))
dummy.model <- dummyVars(formula=f,data=churn.df,fullRank=TRUE)
enc.data <- as.data.frame(predict(dummy.model, churn.df))
churn.df <- cbind(churn.df[ , !(names(churn.df) %in% one_hot_enc_cols)], enc.data)

ordinal_cols = c('tenure','products_number','churn')

for (col in ordinal_cols) {
  churn.df[[col]] <- as.integer(factor(churn.df[[col]],
                                       levels=sort(unique(churn.df[[col]])),
                                       ordered=TRUE))
}

impute_outliers <- function(data, col) {
  Q1 <- quantile(data[[col]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[col]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  median_val <- median(data[[col]][data[[col]] >= lower & data[[col]] <= upper], na.rm = TRUE)
  data[[col]][data[[col]] < lower | data[[col]] > upper] <- median_val
  return(data)
}

churn.df <- impute_outliers(churn.df,'age')
churn.df <- impute_outliers(churn.df,'credit_score')

# Removing duplicate columns
churn.df <- churn.df[, !duplicated(colnames(churn.df))]

ggplotly(ggplot(churn.df,aes(x=age)) + geom_histogram(alpha=0.6,bins=30,color='black') + theme_bw())
ggplotly(ggplot(churn.df,aes(x=credit_score)) + geom_histogram(alpha=0.6,bins=30,color='black') + theme_bw())

# Feature Selection
corr_matrix <- cor(churn.df[,c('estimated_salary','credit_score','age','balance')],use='complete.obs')
corrplot(corr_matrix,method='color',type='full',addCoef.col='black')

cat_cols <- churn.df %>% select_if(is.factor) %>% colnames

for (col in cat_cols) {
  tbl <- table(churn.df[[col]],churn.df$churn)
  print(glue("{col} p-value: ", chisq.test(tbl)$p.value))
}

# The categorical features - country, gender and active_member are statistically significant for the target variable churn.
churn.df <- churn.df %>% dplyr::select(-customer_id)

missmap(churn.df,main='Missingness Map',col=c('yellow','black'),y.at=0,y.labels=NULL)

print(str(churn.df))

# Balancing imbalanced target label churn using SMOTE
churn.df$churn <- as.numeric(as.character(churn.df$churn))

X <- churn.df[,setdiff(colnames(churn.df),"churn")]
y <- churn.df$churn

smote <- SMOTE(X,y,K=5)
churn.df <- smote$data

churn.df <- churn.df %>% rename(churn=class)
levels(churn.df) <- c("0","1")
churn.df$churn <- as.factor(ifelse(churn.df$churn == "2", 1, 0))
churn.df$churn <- as.numeric(as.character(churn.df$churn))

# Feature scaling
mins <- apply(churn.df,2,min)
maxs <- apply(churn.df,2,max)

scaled.df <- scale(churn.df,center=mins,scale=(maxs-mins))
scaled.df <- as.data.frame(scaled.df)
print(head(scaled.df))
print(str(scaled.df))

churn.df$churn <- factor(churn.df$churn, levels = c(0,1), labels = c("0", "1"))

# Train test split
set.seed(101)
split <- sample.split(Y=scaled.df,SplitRatio=0.7)

train <- subset(scaled.df,split==T)
test <- subset(scaled.df,split==F)

# Create a dataframe to store model results
model.performance <- data.frame(
  Model = character(),
  Accuracy = numeric(),
  stringsAsFactors = FALSE
)

# Logistic Regression model
lr <- glm(formula=churn ~ .,family=binomial(link='logit'),data=train)
print(summary(lr))

lr.predictions <- predict(lr,newdata=test,type='response')
lr.predictions <- ifelse(lr.predictions > 0.5, 1, 0)
misclassification.rate <- mean(test$churn != lr.predictions)
print(paste("Logistic Regression Accuracy: ", 1 - misclassification.rate))

accuracy.lr <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "Logistic Regression", Accuracy = accuracy.lr))

table(test$churn,lr.predictions)

conf.matrix <- confusionMatrix(
  data = as.factor(lr.predictions),
  reference = as.factor(test$churn),
  positive = "1" 
)
print(conf.matrix)

# Probit Regression
probit.model <- glm(formula=churn ~ ., data=train, family=binomial(link="probit"))
print(summary(probit.model))

probit.predictions <- predict(probit.model, newdata=test, type='response')
probit.predictions <- ifelse(probit.predictions > 0.5, 1, 0)
table(test$churn,probit.predictions)

misclassification.rate <- mean(test$churn != probit.predictions)
print(paste("Probit Regression Accuracy: ", 1 - misclassification.rate))

accuracy.probit <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "Probit Regression", Accuracy = accuracy.probit))

conf.matrix <- confusionMatrix(
  data = as.factor(probit.predictions),
  reference = as.factor(test$churn),
  positive = "1" 
)
print(conf.matrix)

# Multinomial Logistic Regression
mlr <- multinom(formula=churn ~ ., data=train)
print(summary(mlr))

mlr.predictions <- predict(mlr, newdata=test)

misclassification.rate <- mean(test$churn != mlr.predictions)
print(paste("Multinomial Logistic Regression Accuracy: ", 1 - misclassification.rate))

accuracy.mlr <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "Multinomial Logistic Regression", Accuracy = accuracy.mlr))

conf.matrix <- confusionMatrix(
  data = as.factor(mlr.predictions),
  reference = as.factor(test$churn),
  positive = "1" 
)
print(conf.matrix)

# Lasso Regularization model
x_train <- model.matrix(churn ~ .,data=train)[,-1]
y_train <- as.numeric(as.character(train$churn))
x_test <- model.matrix(churn ~ .,data=test)[,-1]
y_test <- as.numeric(as.character(test$churn))

lasso.model <- cv.glmnet(x_train,y_train,family='binomial',alpha=1)
lasso.predictions <- as.vector(as.integer(predict(lasso.model, x_test, s='lambda.min', type='class')))

misclassification.rate <- mean(y_test != lasso.predictions)
print(paste("Lasso Accuracy: ", 1 - misclassification.rate))

accuracy.lasso <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "Lasso", Accuracy = accuracy.lasso))

conf.matrix <- confusionMatrix(
  data=as.factor(lasso.predictions),
  reference=as.factor(y_test),
  positive='1'
)
print(conf.matrix)

# Ridge regularization model
ridge.model <- cv.glmnet(x_train,y_train,family='binomial',alpha=0)
ridge.predictions <- as.vector(as.integer(predict(ridge.model, x_test, s='lambda.min', type='class')))

misclassification.rate <- mean(y_test != ridge.predictions)
print(paste("Ridge Accuracy: ", 1 - misclassification.rate))

accuracy_ridge <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "Ridge", Accuracy = accuracy_ridge))

conf.matrix <- confusionMatrix(
  data=as.factor(ridge.predictions),
  reference=as.factor(y_test),
  positive='1'
)
print(conf.matrix)

y_train <- factor(ifelse(train$churn == 1, "Yes", "No"))
y_test  <- factor(ifelse(test$churn == 1, "Yes", "No"))

# Create training control
train.control <- trainControl(method='cv',number=5,classProbs=T,summaryFunction=twoClassSummary)

lasso.caret <- caret::train(
  x = x_train,
  y = as.factor(y_train),
  method = 'glmnet',
  trControl = train.control,
  tuneLength = 10
)

alpha <- as.double(lasso.caret$bestTune$lambda)
lambda <- as.double(lasso.caret$bestTune$lambda)

# Optimized regularization model
tuned.reg.model <- glmnet(x_train,y_train,family='binomial',alpha=alpha,lambda=lambda)
tuned.reg.preds <- as.factor(predict(tuned.reg.model, x_test, s='lambda.min', type='class'))

misclassification.rate <- mean(y_test != tuned.reg.preds)
print(paste("Lasso Accuracy: ", 1 - misclassification.rate))

conf.matrix <- confusionMatrix(
  data=as.factor(tuned.reg.preds),
  reference=as.factor(y_test),
  positive='Yes'
)
print(conf.matrix)

# Linear Discriminant Analysis (LDA)
lda <- lda(formula=churn ~ ., data=train)
print(summary(lda))

lda.predictions <- predict(lda, newdata=test, type='response')$class
table(test$churn, lda.predictions)

misclassification.rate <- mean(test$churn != lda.predictions)
print(paste("LDA Accuracy: ", misclassification.rate))

accuracy.lda <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "LDA", Accuracy = accuracy.lda))

conf.matrix <- confusionMatrix(
  data = as.factor(lda.predictions),
  reference = as.factor(test$churn),
  positive = "1" 
)
print(conf.matrix)

# Quadratic Discriminant Analysis (QDA)
qda <- qda(formula=churn ~ ., data=train)
print(summary(qda))

qda.predictions <- predict(qda, newdata=test, type='response')$class
table(test$churn, qda.predictions)

misclassification.rate <- mean(test$churn != qda.predictions)
print(paste("QDA Accuracy: ", 1 - misclassification.rate))

accuracy.qda <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "QDA", Accuracy = accuracy.qda))

conf.matrix <- confusionMatrix(
  data=as.factor(qda.predictions),
  reference=as.factor(test$churn),
  positive='1'
)
print(conf.matrix)

# K Nearest Neighbors (KNN)
knn.predictions <- knn(train,test,train$churn,k=2)
table(knn.predictions,test$churn)

misclassification.rate <- mean(test$churn != knn.predictions)
print(paste("KNN Accuracy: ", 1 - misclassification.rate))

accuracy.knn <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "KNN", Accuracy = accuracy.knn))

conf.matrix <- confusionMatrix(
  data=as.factor(knn.predictions),
  reference=as.factor(test$churn),
  positive="1"
)

print(conf.matrix)

# Since the KNN model achieves an incredible 100% accuracy, the most optimal value of k is 2 which makes sense as we are trying to distinguish between churn and non-churn customers.

train$churn <- factor(train$churn)
test$churn <- factor(test$churn)

# Support Vector Machines (SVM)
svm.model <- svm(formula=churn ~ ., data=train)
print(summary(svm.model))

svm.predictions <- predict(svm.model,newdata=test[,setdiff(names(test),"churn")])
print(table(svm.predictions))

misclassification.rate <- mean(test$churn != svm.predictions)
print(paste("SVM Accuracy: ", 1 - misclassification.rate))

accuracy.svm <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "SVM", Accuracy = accuracy.svm))

conf.matrix <- confusionMatrix(
  data=as.factor(svm.predictions),
  reference=as.factor(test$churn),
  positive='1'
)
print(conf.matrix)

# Create train test split for unscaled data
set.seed(101)
split <- sample.split(Y=churn.df$churn,SplitRatio=0.7)

unscaled.train <- subset(churn.df,split==T)
unscaled.test <- subset(churn.df,split==F)

# Decision Tree
dt.model <- rpart(formula=churn ~ ., data=unscaled.train, method='class')
print(summary(dt.model))

printcp(dt.model)
# plot(dt.model,uniform=T,main='Decision Tree')
# text(dt.model,use.n=T,all=T)

# Plot decision tree model
prp(dt.model)

dt.probabilities <- predict(dt.model,newdata=unscaled.test)
print(head(dt.probabilities))

dt.predictions <- colnames(dt.probabilities)[max.col(dt.probabilities,ties.method='first')]
table(unscaled.test$churn,dt.predictions)

misclassification.rate <- mean(unscaled.test$churn != dt.predictions)
print(paste("Decision Tree Accuracy: ", 1 - misclassification.rate))

accuracy.dt <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "Decision Tree", Accuracy = accuracy.dt))

conf.matrix <- confusionMatrix(
  data=as.factor(dt.predictions),
  reference=as.factor(unscaled.test$churn),
  positive='1'
)
print(conf.matrix)

# Random Forest model
rf.model <- randomForest(formula=churn ~ ., data=unscaled.train)
print(summary(rf.model))

rf.predictions <- predict(rf.model,newdata=unscaled.test)
table(unscaled.test$churn,rf.predictions)

misclassification.rate <- mean(unscaled.test$churn != rf.predictions)
print(paste("Random Forest Accuracy: ", 1 - misclassification.rate))

accuracy.rf <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "Random Forest", Accuracy = accuracy.rf))

conf.matrix <- confusionMatrix(
  data=as.factor(rf.predictions),
  reference=as.factor(unscaled.test$churn),
  positive='1'
)
print(conf.matrix)

# Naive Bayes
nb <- naiveBayes(formula=churn ~ ., data=unscaled.train)
print(summary(nb))

nb.predictions <- predict(nb, newdata=unscaled.test)

table(unscaled.test$churn,nb.predictions)

misclassification.rate <- mean(unscaled.test$churn != nb.predictions)
print(paste("Naive Bayes Accuracy: ", 1 - misclassification.rate))

accuracy.nb <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "Naive Bayes", Accuracy = accuracy.nb))

conf.matrix <- confusionMatrix(
  data=as.factor(nb.predictions),
  reference=as.factor(unscaled.test$churn),
  positive="1"
)
print(conf.matrix)

# C5.0 model
c5.0 <- C5.0(formula=churn ~ .,data=unscaled.train)
print(summary(c5.0))

c5.0.predictions <- predict(c5.0,newdata=unscaled.test)
table(unscaled.test$churn,c5.0.predictions)

misclassification.rate <- mean(unscaled.test$churn != c5.0.predictions)
print(paste("C5.0 Accuracy: ", 1 - misclassification.rate))

accuracy_c5 <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "C5.0", Accuracy = accuracy_c5))

conf.matrix <- confusionMatrix(unscaled.test$churn,c5.0.predictions)
print(conf.matrix)

# Gradient Boosted Trees
train$churn <- as.numeric(as.character(train$churn))
test$churn <- as.numeric(as.character(test$churn))

gbt <- gbm(formula=churn ~ ., data=train, distribution='bernoulli', n.trees = 100, interaction.depth = 3, shrinkage = 0.05)
print(summary(gbt))

gbt.predictions <- predict(gbt, newdata=test, type='response')
print(head(gbt.predictions))
gbt.predictions <- ifelse(gbt.predictions > 0.5, 1, 0)
table(gbt.predictions)

table(test$churn,gbt.predictions)

misclassification.rate <- mean(test$churn != gbt.predictions)
print(paste("Gradient Boosted Trees Accuracy: ", 1 - misclassification.rate))

accuracy.gbt <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "Gradient Boosted Trees", Accuracy = accuracy.gbt))

conf.matrix <- confusionMatrix(
  data=as.factor(gbt.predictions),
  reference=as.factor(test$churn),
  positive='1'
)
print(conf.matrix)

# eXtreme Gradient Boosting (XGBoost)
train$churn <- as.integer(train$churn > 0)
test$churn  <- as.integer(test$churn > 0)
dmatrix.train <- xgb.DMatrix(
  data = as.matrix(train[, setdiff(names(train), "churn")]),
  label = train$churn
)

dmatrix.test <- xgb.DMatrix(
  data = as.matrix(test[, setdiff(names(test), "churn")]),
  label = test$churn
)

xgb.model <- xgboost(data=dmatrix.train,max_depth=4,eta=0.1,nrounds=100,objective='binary:logistic',verbose=0)
print(summary(xgb.model))

xgb.predictions <- predict(xgb.model,newdata=dmatrix.test,type='response')
print(head(xgb.predictions))

xgb.predictions <- ifelse(xgb.predictions > 0.5, 1, 0)
table(test$churn,xgb.predictions)

misclassification.rate <- mean(test$churn != xgb.predictions)
print(paste("XGBoost Accuracy: ", 1 - misclassification.rate))

accuracy.xgb <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "XGBoost", Accuracy = accuracy.xgb))

conf.matrix <- confusionMatrix(
  data=as.factor(xgb.predictions),
  reference=as.factor(test$churn),
  positive='1'
)
print(conf.matrix)

# Adaptive Boosting model
train$churn <- factor(as.character(train$churn))
test$churn <- factor(as.character(test$churn))

adb <- boosting(formula=churn ~ ., data=train,mfinal=100,boos=TRUE)
print(summary(adb))

adb.predictions <- predict(adb,newdata=test)$class
table(test$churn,adb.predictions)

misclassification.rate <- mean(test$churn != adb.predictions)
print(paste("Adaptive Boosting Accuracy: ", 1 - misclassification.rate))

accuracy.adb <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "Adaptive Boosting", Accuracy = accuracy.adb))

# Bagging
bag <- bagging(formula=churn ~ ., data=train,nbagg=25)
print(summary(bag))

bag.predictions <- predict(bag,newdata=test)$class
table(test$churn,bag.predictions)

misclassification.rate <- mean(test$churn != bag.predictions)
print(paste("Bagging Accuracy: ", 1 - misclassification.rate))

accuracy.bag <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "Bagging", Accuracy = accuracy.bag))

conf.matrix <- confusionMatrix(
  data=as.factor(bag.predictions),
  reference=as.factor(test$churn),
  positive='1'
)
print(conf.matrix)

# Hyperparameter Tuning & Cross Validation
unscaled.train$churn <- factor(unscaled.train$churn, levels = c(0, 1), labels = c("No", "Yes"))
unscaled.test$churn  <- factor(unscaled.test$churn,  levels = c(0, 1), labels = c("No", "Yes"))
train$churn <- factor(train$churn, levels = c(0,1), labels = c("No", "Yes"))
test$churn <- factor(test$churn, levels = c(0,1), labels = c("No", "Yes"))

# Random Forest
tuned.rf <- caret::train(
  churn ~ ., 
  data=unscaled.train,
  method="rf",
  trControl=train.control,
  tuneGrid=expand.grid(mtry=c(2,4,6,8)),
  ntree=200
)
print(tuned.rf)
print(summary(tuned.rf))

tuned.rf.model <- randomForest(formula=churn ~ .,data=unscaled.train,mtry=4,ntree=200)
tuned.rf.predictions <- predict(tuned.rf.model,newdata=unscaled.test)
table(unscaled.test$churn,tuned.rf.predictions)

misclassification.rate <- mean(unscaled.test$churn != tuned.rf.predictions)
print(paste("Tuned Random Forest Accuracy: ", 1 - misclassification.rate))

accuracy.tuned.rf <- 1 - misclassification.rate
model.performance <- rbind(model.performance,data.frame(Model="Tuned Random Forest", Accuracy=accuracy.tuned.rf))

conf.matrix <- confusionMatrix(
  data=as.factor(tuned.rf.predictions),
  reference=as.factor(unscaled.test$churn),
  positive='Yes'
)
print(conf.matrix)

# Gradient Boosted Trees
gbm_grid <- expand.grid(
  interaction.depth = c(1, 3, 5),
  n.trees = c(100, 200),
  shrinkage = c(0.05, 0.1),
  n.minobsinnode = c(10, 20)
)

tuned.gbm <- caret::train(
  churn ~ ., 
  data=train,
  method="gbm",
  metric="accuracy",
  trControl=train.control,
  tuneGrid=gbm_grid,
  verbose=FALSE
)
print(tuned.gbm)
print(summary(tuned.gbm))

train$churn <- as.numeric(ifelse(train$churn == "Yes",1,0))
test$churn <- as.numeric(ifelse(test$churn == "Yes",1,0))

tuned.gbm.model <- gbm(formula=churn ~ .,distribution='bernoulli',data=train,n.trees=200,interaction.depth=5,shrinkage=0.1,n.minobsinnode=20)
tuned.gbm.predictions <- predict(tuned.gbm.model,newdata=test,type='response')
tuned.gbm.predictions <- ifelse(tuned.gbm.predictions > 0.5, 1, 0)
table(test$churn,tuned.gbm.predictions)

misclassification.rate <- mean(test$churn != tuned.gbm.predictions)
print(paste("Tuned GBT Accuracy: ", 1 - misclassification.rate))

accuracy.tuned.gbt <- 1 - misclassification.rate
model.performance <- rbind(model.performance,data.frame(Model="Tuned GBT", Accuracy=accuracy.tuned.gbt))

conf.matrix <- confusionMatrix(
  data=as.factor(tuned.gbm.predictions),
  reference=as.factor(test$churn),
  positive='1'
)
print(conf.matrix)

# C5.0 
train$churn <- factor(train$churn, levels = c(0, 1), labels = c("No", "Yes"))
test$churn <- factor(test$churn, levels = c(0, 1), labels = c("No", "Yes"))

c50_grid <- expand.grid(
  trials = c(10, 50, 100),    # boosting iterations
  model = c("tree", "rules"), # tree or rule-based
  winnow = c(TRUE, FALSE)
)

tuned.c50 <- caret::train(
  churn ~ ., 
  data=train,
  method="C5.0",
  metric="accuracy",
  trControl=train.control,
  tuneGrid=c50_grid
)
print(tuned.c50)
print(summary(tuned.c50))

tuned.c50.model <- C5.0(formula=churn ~ .,data=train,trials=100,model='rules',winnow=F)
tuned.c50.predictions <- predict(tuned.c50.model,newdata=test,type='class')
table(test$churn,tuned.c50.predictions)

misclassification.rate <- mean(test$churn != tuned.c50.predictions)
print(paste("Tuned C5.0 Accuracy: ", 1 - misclassification.rate))

accuracy.tuned.c5.0 <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model="Tuned C5.0", Accuracy=accuracy.tuned.c5.0))

conf.matrix <- confusionMatrix(
  data=as.factor(tuned.c50.predictions),
  reference=as.factor(test$churn),
  positive='Yes'
)
print(conf.matrix)

# Naive Bayes
nb.grid <- expand.grid(
  laplace = c(0, 0.5, 1),     # Laplace smoothing
  usekernel = c(TRUE, FALSE),
  adjust = c(0, 1)
)

tuned.nb <- caret::train(
  churn ~ ., 
  data=train,
  method="naive_bayes",
  metric="accuracy",
  trControl=train.control,
  tuneGrid=nb.grid
)
print(tuned.nb)
print(summary(tuned.nb))

tuned.nb.model <- naiveBayes(formula=churn ~ .,data=train,laplace=0,usekernel=T,adjust=1)
tuned.nb.predictions <- predict(tuned.nb.model,newdata=test,type='class')
table(test$churn, tuned.nb.predictions)

misclassification.rate <- mean(test$churn != tuned.nb.predictions)
print(paste("Tuned Naive Bayes Accuracy: ", 1 - misclassification.rate))

accuracy.tuned.nb <- 1 - misclassification.rate
model.performance <- rbind(model.performance,data.frame(Model="Tuned Naive Bayes", Accuracy=accuracy.tuned.nb))

conf.matrix <- confusionMatrix(
  data=as.factor(tuned.nb.predictions),
  reference=as.factor(test$churn),
  positive='Yes'
)
print(conf.matrix)

# Decision Tree
dt.grid <- expand.grid(cp = seq(0.001, 0.05, length=5))  # complexity parameter

tuned.dt <- caret::train(
  churn ~ ., 
  data=train,
  method="rpart",
  metric="ROC",
  trControl=train.control,
  tuneGrid=dt.grid
)
print(tuned.dt)
print(summary(tuned.dt))

tuned.dt <- rpart(formula=churn ~ .,data=train,method='class',cp=0.001)
tuned.dt.predictions <- predict(tuned.dt,newdata=test,type='class')
table(test$churn,tuned.dt.predictions)

misclassification.rate <- mean(test$churn != tuned.dt.predictions)
print(paste("Tuned Decision Tree Accuracy: ", 1 - misclassification.rate))

accuracy.tuned.dt <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model="Tuned Decision Tree", Accuracy=accuracy.tuned.dt))

conf.matrix <- confusionMatrix(
  data=as.factor(tuned.dt.predictions),
  reference=as.factor(test$churn),
  positive='Yes'
)
print(conf.matrix)

# Train test split
set.seed(101)
split <- sample.split(Y=scaled.df,SplitRatio=0.7)

train <- subset(scaled.df,split==T)
test <- subset(scaled.df,split==F)

# Neural network using Keras
train$churn <- as.integer(train$churn) 
test$churn <- as.integer(test$churn)
X_train <- as.matrix(train[, setdiff(colnames(train), "churn")])
X_test  <- as.matrix(test[, setdiff(colnames(test), "churn")])
y_train <- keras::to_categorical(train$churn,num_classes=2)
y_test <- keras::to_categorical(test$churn,num_classes=2)

# Define the model architecture
model <- keras_model_sequential() %>%
  layer_dense(units=16,activation='relu',input_shape=ncol(X_train)) %>%
  layer_dense(units=8,activation='relu') %>%
  layer_dense(units=2,activation='sigmoid')

# Compile the model
model %>% compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics='accuracy'
)

# Train the model
history <- model %>% fit(X_train,y_train,epochs=50,batch_size=32,validation_data=list(X_test,y_test))

# Evaluate the model
model %>% evaluate(X_test,y_test)

predictions <- model %>% predict(X_test)
print(head(predictions))
predicted.classes <- ifelse(predictions[,2] > 0.5, 1, 0)
y_test_actual <- apply(y_test,1,which.max) - 1
misclassification.rate <- mean(y_test_actual != predicted.classes)
print(paste("Keras Neural Network Accuracy: ", 1 - misclassification.rate))

accuracy.keras <- 1 - misclassification.rate
model.performance <- rbind(model.performance, data.frame(Model = "Neural Network (Keras)", Accuracy = accuracy.keras))

conf.matrix <- confusionMatrix(
  data=as.factor(predicted.classes),
  reference=as.factor(y_test_actual),
  positive='1'
)
print(conf.matrix)

# Compare all models and save the best one
print(model.performance %>% arrange(desc(Accuracy)))
best_model <- model.performance[which.max(model.performance$Accuracy),]
print(paste("Best model is:", best_model$Model, "with accuracy:", best_model$Accuracy))

control <- trainControl(method = "cv", number = 5)

train$churn <- factor(train$churn, levels = c(0, 1), labels = c("No", "Yes"))
test$churn <- factor(test$churn, levels = c(0, 1), labels = c("No", "Yes"))

knn.model <- caret::train(
  churn ~ ., 
  data=train,
  method="knn",
  trControl=control,
  tuneGrid=expand.grid(k=c(2))  # optimal k
)

saveRDS(xgb.model, "customer_churn_classifier.rds")

loaded.model <- readRDS('customer_churn_classifier.rds')
predictions <- predict(loaded.model, newdata=dmatrix.test, type='class')
predictions <- as.factor(ifelse(predictions > 0.5, "Yes", "No"))
table(test$chur, predictions)

misclassification.rate <- mean(test$churn != predictions)
print(paste("Loaded XGBoost Model Accuracy: ", 1 - misclassification.rate))

conf.matrix <- confusionMatrix(
  data=as.factor(predictions),
  reference=as.factor(test$churn),
  positive='Yes'
)
print(conf.matrix)