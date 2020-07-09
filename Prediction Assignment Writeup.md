Prediction Assignment Writeup
Alvaro Vega
8/7/2020
Prediction Assignment Writeup
Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

Goal
The goal of your project is to predict the manner in which they did the exercise. This is the “classe” variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

Installing Packages
install.packages('caret')
install.packages('rpart')
install.packages("ipred")
install.packages("randomForest")
Loading libraries
library(caret)
library(rpart)
library(rpart.plot)
library(ipred)
library(randomForest)
Loading the Data
training  <- read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv')
testing <- read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv')
#convert classe to factors
training$classe <- as.factor(training$classe)
#head(training,3)
dim(training)
## [1] 19622   160
#head(testing,3)
dim(testing)
## [1]  20 160
Cleaning the data
Cleaning the training data
# remove variables with nearly zero variance
nzv <- nearZeroVar(training)
training <- training[, -nzv]

# remove variables that are almost always NA
remove_NA <- sapply(training, function(x) mean(is.na(x))) > 0.95
training <- training[, remove_NA==F]

# remove first 5 useless variables 
training <- training[, -(1:5)]
# New size of training data
dim(training)
## [1] 19622    54
Cleaning the testing data
# remove variables with nearly zero variance
nzv <- nearZeroVar(testing)
testing <- testing[, -nzv]

# remove variables that are almost always NA
remove_NA <- sapply(testing, function(x) mean(is.na(x))) > 0.95
testing <- testing[, remove_NA==F]

# remove first 5 useless variables 
testing <- testing[, -(1:5)]
# New size of testing data
dim(testing)
## [1] 20 54
Partitioning Training Data
inTrain <- createDataPartition(training$classe, p=0.7, list=FALSE)
myTraining <- training[inTrain, ]
myTesting <- training[-inTrain, ]
dim(myTraining)
## [1] 13737    54
dim(myTesting)
## [1] 5885   54
Creating the Models
Tree Model
# Create the tree model
dt <- rpart(formula = classe ~ ., 
                        data = myTraining, 
                        method = "class")
# Generate predicted classes using the dt model
dt_prediction <- predict(object = dt,  
                          newdata = myTesting,   
                          type = "class",
                          parms = list(split = "gini"))  
# Calculate the confusion matrix for the test set
confusionMatrix(data =dt_prediction,       
                  reference = myTesting$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1522  178   21   50   19
##          B   39  672   85   44   28
##          C    0   60  795   32    2
##          D   97  197  115  755  175
##          E   16   32   10   83  858
## 
## Overall Statistics
##                                           
##                Accuracy : 0.782           
##                  95% CI : (0.7712, 0.7925)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7241          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9092   0.5900   0.7749   0.7832   0.7930
## Specificity            0.9364   0.9587   0.9807   0.8813   0.9706
## Pos Pred Value         0.8503   0.7742   0.8943   0.5639   0.8589
## Neg Pred Value         0.9629   0.9069   0.9538   0.9540   0.9542
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2586   0.1142   0.1351   0.1283   0.1458
## Detection Prevalence   0.3042   0.1475   0.1511   0.2275   0.1698
## Balanced Accuracy      0.9228   0.7743   0.8778   0.8323   0.8818
# Create the Bagging model
b_model <- bagging(formula = classe ~ ., 
                        data = myTraining,
                        coob = TRUE)

# Print the model
print(b_model)
## 
## Bagging classification trees with 25 bootstrap replications 
## 
## Call: bagging.data.frame(formula = classe ~ ., data = myTraining, coob = TRUE)
## 
## Out-of-bag estimate of misclassification error:  0.0046
# Generate predicted classes using the model object
b_prediction <- predict(object = b_model,    
                              newdata = myTesting,  
                            type = "class")  # return classification labels

# Calculate the confusion matrix for the test set
confusionMatrix(data = b_prediction ,       
                  reference = myTesting$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    5    0    0    0
##          B    0 1131    3    0    2
##          C    0    2 1022    1    0
##          D    0    1    1  962    4
##          E    2    0    0    1 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9963          
##                  95% CI : (0.9943, 0.9977)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9953          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9930   0.9961   0.9979   0.9945
## Specificity            0.9988   0.9989   0.9994   0.9988   0.9994
## Pos Pred Value         0.9970   0.9956   0.9971   0.9938   0.9972
## Neg Pred Value         0.9995   0.9983   0.9992   0.9996   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1922   0.1737   0.1635   0.1828
## Detection Prevalence   0.2850   0.1930   0.1742   0.1645   0.1833
## Balanced Accuracy      0.9988   0.9960   0.9977   0.9984   0.9969
#Create the random forest model
rf_model <- randomForest(formula = classe ~ ., 
                               data = myTraining)
# Print the model output
print(rf_model)
## 
## Call:
##  randomForest(formula = classe ~ ., data = myTraining) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.23%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3906    0    0    0    0 0.000000000
## B    5 2653    0    0    0 0.001881114
## C    0    6 2390    0    0 0.002504174
## D    0    0   14 2237    1 0.006660746
## E    0    0    0    5 2520 0.001980198
# Generate predicted classes using the random forest
rf_prediction <- predict(object = rf_model,   # model object 
                              newdata = myTesting,  # test dataset
                              type = "class") # return classification labels
                            
# Calculate the confusion matrix for the random forest
confusionMatrix(data =rf_prediction,       # predicted classes
                        reference = myTesting$classe)  # actual classes
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    2    0    0    0
##          B    1 1136    5    0    0
##          C    0    1 1021    1    0
##          D    0    0    0  963    4
##          E    0    0    0    0 1078
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9976         
##                  95% CI : (0.996, 0.9987)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.997          
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9974   0.9951   0.9990   0.9963
## Specificity            0.9995   0.9987   0.9996   0.9992   1.0000
## Pos Pred Value         0.9988   0.9947   0.9980   0.9959   1.0000
## Neg Pred Value         0.9998   0.9994   0.9990   0.9998   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1930   0.1735   0.1636   0.1832
## Detection Prevalence   0.2846   0.1941   0.1738   0.1643   0.1832
## Balanced Accuracy      0.9995   0.9981   0.9974   0.9991   0.9982
Accuracy:
Tree Model: 0.7373
Bagging Model: 0.9964
Random Forest Model: 0.9978
# Random Forest Model yields very good results 99.78% accuracy!
Predicting on the Testing Data
Prediction on the Bagging model
# Generate predicted classes using the bagging
b_prediction <- predict(object = b_model,    
                              newdata = testing,  
                            type = "class")  # return classification labels
b_prediction
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
Prediction on the random forest model
# Generate predicted classes using the model object
rf_prediction <- predict(object = rf_model,   # model object 
                              newdata = testing,  # test dataset
                              type = "class") # return classification labels
rf_prediction
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
