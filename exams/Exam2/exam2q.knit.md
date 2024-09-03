---
title: "Exam 2: Boosting Models"
format: html
editor: visual
---




-------

Sample calculations can be found here: [Link](Exam2_sol.html)

#### Problem 1. Adaboost.  
Given the data.

<center>

|    |   $x_1$ |   $x_2$ |   $y$ |
|:----|:------|:------|:-----|
|  0 |    1 |    2 |   1 |
|  1 |    2 |    5 |   1 |
|  2 |    3 |    4 |  -1 |
|  3 |    4 |    0 |  -1 |
|  4 |    0 |    1 |  -1 |
</center>

Suppose we use this data to train an adaboost model with the learning rate $L=1$ and obtain the three stumps as follows.

  - Stump 1: $I(x_1<2.5)$
  - Stump 2: $I(x_2>1.5)$
  - Stump 3: $I(x_2>4.5)$

  a. Compute the weight of the data in the first round.
  
  b. After the first round, which observations should have the weights increased? Which observations should have the weights decreased? Explain why. 
  
  c. Compute the weights of the data in the second round and third round. 
  
  d. Compute the voting powers of the three stumps. Which stump has the highest voting power? 
  
  e. Draw the decision boundary of the adaboost. This video shows how to make decision boundary of the adaboost: [link](https://bryant.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=e0a751e8-d6c8-4a7d-98f7-adcc01064682)
  
        Class note for drawing decision boundary of Adaboost can be found here: [Link](adaboost_db.pdf)
  
  f. Compute the error (misclassification) of the adataboost. 

<br>

#### Problem 2. Gradient Boosting

Given the data.

<center>

| X | y | 
|:---|:---|
| -1 | 1 |
| 2 | 2 |
| 4 | 5 |

</center>

We will train a gradient boosted tree (stump) on this data with the learning rate of 1.

  a. Calculate the prediction of the first round. 
  
  b. Calculate the target of the second round
  
  c. Suppose that the stump in the second round is: $X<3$. Calculate the predictions of the second round. What do the second round predict?
  
  d.  What are the final predictions of the gradient boosting model for $y$ in the training data after the second round?
  
  e. Calculate the target of the third round. 
  
  f. Suppose that the stump in the third round is: $X<0.5$. Calculate the predictions of the third round. What do the third round predict?
  
  g. What are the final predictions of the gradient boosting model for $y$ in the training data after the third round?
  

