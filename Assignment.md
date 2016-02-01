# load libraries
library(caret)
## Loading required package: lattice
## Loading required package: ggplot2
# load the training set
pml.training <- read.csv("pml-training.csv", na.strings = c("NA", ""))

# load the testing set Note: the testing set is not used in this analysis
# the set is only used for the second part of the assignment when the model
# is used to predict the classes
pml.testing <- read.csv("pml-testing.csv", na.strings = c("NA", ""))

# summary(pml.training)
We are interested in variables that predict the movement The set contains a number of variables that can be removed:

X (= row number)
user_name (= the name of the subject)
cvtd_timestamp is removed because it is a factor instead of a numeric value and the raw_timestamp_part_1 + raw_timestamp_part_2 contain the same info in numeric format.

rIndex <- grep("X|user_name|cvtd_timestamp", names(pml.training))
pml.training <- pml.training[, -rIndex]
Some variable have near Zero variance which indicates that they do not contribute (enough) to the model. They are removed from the set.

nzv <- nearZeroVar(pml.training)
pml.training <- pml.training[, -nzv]
A number of variable contain (a lot of) NA's. Leaving them in the set not only makes the model creation slower, but also results in lower accuracy in the model. These variables will be removed from the set:

NAs <- apply(pml.training, 2, function(x) {
    sum(is.na(x))
})
pml.training <- pml.training[, which(NAs == 0)]
The original set is rather large (19622 obs. of 56 variables). We create a smaller training set of 80% of the original set

tIndex <- createDataPartition(y = pml.training$classe, p = 0.2, list = FALSE)
pml.sub.training <- pml.training[tIndex, ]  # 3927 obs. of 56 variables
pml.test.training <- pml.training[-tIndex, ]  # test set for cross validation
Model creation

We can now create a model based on the pre-processed data set. Note that at this point, we are still working with a large set of variables. We do have however a reduced number of rows.

A first attempt to create a model is done by fitting a single tree:

modFit <- train(pml.sub.training$classe ~ ., data = pml.sub.training, method = "rpart")
## Loading required package: rpart
modFit
## CART 
## 
## 3927 samples
##   55 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 3927, 3927, 3927, 3927, 3927, 3927, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp    Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.04  0.5       0.4    0.09         0.1     
##   0.04  0.5       0.3    0.09         0.1     
##   0.1   0.3       0.05   0.04         0.06    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.04.
results <- modFit$results
round(max(results$Accuracy), 4) * 100
## [1] 52.71
Note that running the train() function can take some time! The accuracy of the model is low: 52.71 %

A second attempt to create a model is done by using Random forests:

ctrl <- trainControl(method = "cv", number = 4, allowParallel = TRUE)
modFit <- train(pml.sub.training$classe ~ ., data = pml.sub.training, method = "rf", 
    prof = TRUE, trControl = ctrl)
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
modFit
## Random Forest 
## 
## 3927 samples
##   55 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 2946, 2945, 2945, 2945 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.006        0.008   
##   30    1         1      0.005        0.007   
##   60    1         1      0.004        0.006   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 28.
results <- modFit$results
round(max(results$Accuracy), 4) * 100
## [1] 98.98
This second attempt provides us with a model that has a much higher accuracy: : 98.98 %

Cross-validation

We now use the modFit to predict new values within the test set that we created for cross-validation:

pred <- predict(modFit, pml.test.training)
pml.test.training$predRight <- pred == pml.test.training$classe
table(pred, pml.test.training$classe)
##     
## pred    A    B    C    D    E
##    A 4461    3    0    0    0
##    B    3 3016   32    0    1
##    C    0   18 2705   36    3
##    D    0    0    0 2536   37
##    E    0    0    0    0 2844
As expected the predictions are not correct in all cases. We can calculate the accuracy of the prediction:

pRes <- postResample(pred, pml.test.training$classe)
pRes
## Accuracy    Kappa 
##   0.9915   0.9893
The prediction fitted the test set even slightly better than the training set: 99.1526 %

Expected out of sample error

We can calculate the expected out of sample error based on the test set that we created for cross-validation:

cfM <- confusionMatrix(pred, pml.test.training$classe)
cfM
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4461    3    0    0    0
##          B    3 3016   32    0    1
##          C    0   18 2705   36    3
##          D    0    0    0 2536   37
##          E    0    0    0    0 2844
## 
## Overall Statistics
##                                        
##                Accuracy : 0.992        
##                  95% CI : (0.99, 0.993)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.989        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.993    0.988    0.986    0.986
## Specificity             1.000    0.997    0.996    0.997    1.000
## Pos Pred Value          0.999    0.988    0.979    0.986    1.000
## Neg Pred Value          1.000    0.998    0.998    0.997    0.997
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.172    0.162    0.181
## Detection Prevalence    0.284    0.194    0.176    0.164    0.181
## Balanced Accuracy       1.000    0.995    0.992    0.992    0.993
The expected out of sample error is: 0.8474 %
