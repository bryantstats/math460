---
title: "Exam 1"
format: pdf
editor: visual
---




*Name:*

## Problem


::: {.cell}

:::


-   Given the **training** data.

<center>


::: {.cell}
::: {.cell-output-display}


| Class| Sex| Survived|
|-----:|---:|--------:|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   1|        1|
|     1|   0|        1|
|     1|   0|        1|
|     1|   0|        1|
|     3|   1|        1|
|     3|   0|        1|
|     2|   1|        0|
|     3|   0|        1|
|     2|   1|        0|
|     2|   1|        0|
|     3|   1|        0|
|     2|   1|        0|
|     3|   0|        0|
|     1|   0|        0|
|     2|   0|        0|
|     1|   1|        0|
|     2|   1|        0|


:::
:::


</center>

------------------------------------------------------------------------

Using Gini Index as the measure for impurity to:

1.  Grow the **maximum tree** [with THREE leaves]{.underline} (a stopping rule!) on the data. Draw the (diagram of) tree.
2.  Find the misclassification rate on training data of the maximal tree
3.  Draw all the possible **subtrees**
4.  Validate the maximal tree and all the subtrees on the following data to select the **optimal tree**. Note: if the chance of Survived is 1/2, predict `Survived`

<center>


::: {.cell}
::: {.cell-output-display}


| Class| Sex| Survived|
|-----:|---:|--------:|
|     1|   0|        1|
|     1|   1|        1|
|     3|   1|        0|
|     2|   0|        0|


:::
:::


</center>

