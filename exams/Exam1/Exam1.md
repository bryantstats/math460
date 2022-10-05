<style>
  .reveal pre {font-size: 12px;}
  body {
    overflow: scroll;
}
</style>

Classification Trees - Practice Problems
========================================================
author: Son Nguyen
date: September 9, 2019
font-family: Garamond


Reading Materials
=======================================================
- Max Kuhn. Chapter 14. Section 14.1 


Problem 1
=======================================================

<center>
![](ct1.png)
</center>

1.  Draw the decision tree with the prediction in the leaves.
3.  Calculate the predicted probabilities produced by this tree.
3.  Calculate the misclassification rate of the tree
4.  Calculate the misclassification rate of the tree on the following validation data

<center>

| $x_1$ | $x_2$ | Target |
|-------|-------|--------|
| 6     | 0     | x      |
| 5     | 7     | o      |
| -3    | 0     | x      |
| 1     | 10    | o      |

</center>

Problem 1.1 - Solution
=======================================================
<center>
![](ct2.png)
</center>


Problem 1.2 - Solution
=======================================================
<center>
![](ct1.png)
</center>

- Predicted Probability is the probability of the target being **positive**. 
- Thus, Predicted Probability of observations in **Leaf 1** is 1/5. 
- Predicted Probability of observations in **Leaf 2** is 2/2 or 1. 
- Predicted Probability of observations in **Leaf 3** is 3/5.  
- Predicted Probability of observations in **leaf 4** is 0. 

Problem 1.3 - Solution
=======================================================
<center>
![](ct1.png)
</center>

- Misclassification rate is 3/13

Problem 1.4 - Solution
=======================================================
- Prediction on the validation data

<center>

| $x_1$ | $x_2$ | Target | Prediction |
|-------|-------|--------|------------|
| 6     | 0     | x      |      x      |
| 5     | 7     | o      |      o      |
| -3    | 0     | x      |      o **(missed)**      |
| 1     | 10    | o      |      x  **(missed)**    |

</center>

- Misclassification Rate is 2/4 = 0.5


Problem 2
=======================================================
Using Gini-Index as a measure if impurity, calculate the impurity gain for the following split.

<center>
![](s17.png)
</center>


Problem 2 Solution
=======================================================

<center>
![](s18.png)
</center>


Problem 3
=======================================================
Decide the best vertical first split for the following dataset.  Impurity can be measured by entropy or Gini-Index.

<center>
![](s15.png)
</center>

Problem 3 - Solution
=======================================================
Decide the best vertical first split for the following dataset.  Impurity can be measured by entropy or Gini-Index.

<center>
![](s16.png)
</center>


Problem 4
=======================================================


- Given the **training** data. Using Gini Index as the measure for impurity to: 

<center>

| Class| Sex| Survived|
|-----:|---:|--------:|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   1|        1|
|     3|   0|        1|
|     2|   1|        1|
|     3|   0|        1|
|     2|   1|        0|
|     2|   1|        0|
|     3|   1|        0|
|     2|   1|        0|
|     3|   0|        0|
|     3|   1|        0|
|     2|   0|        0|
|     1|   1|        0|
|     2|   1|        0|
</center>

1. Grow the **maximum tree** with four leaves (a stopping rule!) on the data.  Draw the (diagram of) tree. 
2. Find the misclassification rate on training data of the maximal tree
3. Draw all the possible **subtrees**
4. Validate the maximal tree and the subtrees on the following data to select the **optimal tree**. Note: if the chance of Survived is 1/2, predict `Survived`

<center>

| Class| Sex| Survived|
|-----:|---:|--------:|
|     1|   0|        1|
|     1|   1|        1|
|     3|   1|        0|
|     2|   0|        0|
</center>


Problem 4.1 Growing a Tree
=======================================================
- We need to find the impurity gain of all possible splits
- What are possible first splits? 

Problem 4.1 Growing a Tree
=======================================================
- What are possible first splits? 
- Split 1:  Split Sex into Male and Female
- Split 2:  Split Class into class 1 and class 2,3 
- Split 3:  Split Class into class 2 and class 1,3 
- Split 4:  Split Class into class 3 and class 1,2 

Total **four** possible splits!

Problem 4.1 Growing a Tree
=======================================================
- Split 1: 
<center>
![](s1.png)

- Impurity Gain: 0.1758

</center>

Problem 4.1 Growing a Tree
=======================================================
- Split 2: 
<center>
![](s2.png)

- Impurity Gain: 0.1879

</center>

Problem 4.1 Growing a Tree
=======================================================
- Split 3: 
<center>
![](s3.png)

- Impurity Gain: 0.1259

</center>

Problem 4.1 Growing a Tree
=======================================================
- Split 4: 
<center>
![](s4.png)

- Impurity Gain: 0.015

</center>


Problem 4.1 Growing a Tree
=======================================================
<center>


| Split | Impurity Gain     |
|-------|-------------------|
| 1     | 0.1758            |
| 2     | 0.1879 (**Best!**) |
| 3     | 0.1259            |
| 4     | 0.015             |

- Thus, split 2 is the first slit of the tree.  
</center>

Problem 4.1 Growing a Tree
=======================================================
- First Split
<center>
![](s21.png)
</center>
- Since the stopping rule is not satisfied, we continue to split. 
- We need to decide the second split. It could be a split at node A or node B


Problem 4.1 Growing a Tree
=======================================================
- Let's look at node A. 
- Data at node A (Class = 1): 

<center>

| Class| Sex| Survived|
|-----:|---:|--------:|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   1|        1|
|     1|   1|        0|
</center>

- What are possible splits at node A?

Problem 4.1 Growing a Tree
=======================================================
<center>

| Class| Sex| Survived|
|-----:|---:|--------:|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   1|        1|
|     1|   1|        0|
</center>

- Node A can only be splitted at Sex. 


Problem 4.1 Growing a Tree
=======================================================
- Now let's look at node B. 
- Data at node A (Class = 2, 3): 

<center>

| Class| Sex| Survived|
|-----:|---:|--------:|
|     3|   0|        1|
|     2|   1|        1|
|     3|   0|        1|
|     2|   1|        0|
|     2|   1|        0|
|     3|   1|        0|
|     2|   1|        0|
|     3|   0|        0|
|     3|   1|        0|
|     2|   0|        0|
|     2|   1|        0|
</center>

- What are possible splits at node B?


Problem 4.1 Growing a Tree
=======================================================
<center>

| Class| Sex| Survived|
|-----:|---:|--------:|
|     3|   0|        1|
|     2|   1|        1|
|     3|   0|        1|
|     2|   1|        0|
|     2|   1|        0|
|     3|   1|        0|
|     2|   1|        0|
|     3|   0|        0|
|     3|   1|        0|
|     2|   0|        0|
|     2|   1|        0|
</center>

- What are possible splits at node B?
- Node B can be splittted at Sex and Class


Problem 4.1 Growing a Tree
=======================================================
- There are **three** candidate for the second split: 
- Split node A at Sex 
- Split node B at Sex  
- Split node B at Class 
- What is the second split? We need to compute the impurity gain for these splits to figure out the best one. 



Problem 4.1 Growing a Tree
=======================================================
- Splitting node A at Sex: 

<center>
![](s5.png)
</center>

- Impurity Gain: 0.0864

Problem 4.1 Growing a Tree
=======================================================
- Splitting node B at Sex: 

<center>
![](s6.png)
</center>

- Impurity Gain: 0.0590

Problem 4.1 Growing a Tree
=======================================================
- Splitting node B at Class: 

<center>
![](s7.png)
</center>

- Impurity Gain: 0.027


Problem 4.1 Growing a Tree
=======================================================
<center>


| Candidate Split | Impurity Gain     |
|-------|-------------------|
| Split node A at Sex      | 0.0864     (**Best!**)       |
| Split node B at Sex      | 0.0590  |
| Split node B at Class      | 0.027            |

- Thus, the second split is splitting node A at Sex
</center>


Problem 4.1 Growing a Tree
=======================================================
- Update the tree

<center>
![](s8.png)
</center>

Problem 4.1 Growing a Tree
=======================================================
<center>
![](s8.png)
</center>
 
- Since the stopping rule is not satisfied, we keep splitting the tree
- What is the third split?

Problem 4.1 Growing a Tree
=======================================================
<center>
![](s8.png)
</center>

- What is the third split?
- The third split cannot be at node C and D because the variable Sex and Class are constant in these nodes.
- The third split has to be in node B
- We already that nodes B can be splitted into way and the one with higher impurity is the split at Sex
- Therefore, the third split is splitting node B at Sex

Problem 4.1 Growing a Tree
=======================================================
- Maximal Tree
<center>
![](s9.png)
</center>
 
- Since the stopping rule is satisfied, splitting process stops.  
- We obtain the **maximal tree**.

Problem 4.2 Misclassification Rate
=======================================================
- Maximal Tree
<center>
![](s9.png)
</center>
 
- The prediction of each leaf is the majority.  Thus, the misclassification of a leaf is the minority in the leaf. 
- Total misclassification is: 0+1+2+1 =4
- Misclassification Rate is 4/20 = 20%

Problem 4.3 Subtrees
=======================================================
- There are three subtrees that can be prunned from the maximal tree


Problem 4.3 Subtrees
=======================================================
- Subtree 1

<center>
![](s10.png)
</center>

Problem 4.3 Subtrees
=======================================================
- Subtree 2

<center>
![](s11.png)
</center>

Problem 4.3 Subtrees
=======================================================
- Subtree 3

<center>
![](s12.png)
</center>


Problem 4.4 Optimal Tree
=======================================================
- Predictions of all the subtrees and the maximal tree

<center>

| Class | Sex | Survived | Subtree1 Predicts | Subtree2 Predicts | Subtree3 Predicts | Maximal Tree Predicts |
|-------|-----|----------|-------------------|-------------------|-------------------|-----------------------|
| 1     | 0   | 1        | 1                 | 1                 | 1                 | 1                     |
| 1     | 1   | 1        | 1                 | 1                 | 1                 | 1                     |
| 3     | 1   | 0        | 0                 | 1 (Missed!)       | 0                 | 1(Missed)             |
| 2     | 0   | 0        | 0                 | 0                 | 0                 | 0                     |


</center>

- We see that Subtree 1 and Subtree 3 both have perfect predictions! (0 misclassification)
- Since Subtree 3 has less leaves than subtree 1, it is simpler than subtree 3. 
- We select Subtree 3 as the optimal tree

Problem 4.4 Winning Model
=======================================================
- Winning Model! 
<center>
![](s12.png)
![](s13.png)
![](s14.png)
</center>
