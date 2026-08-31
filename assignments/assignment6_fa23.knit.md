---
title: "Handwritten Recognition with Ensemble Models"
format: html
editor: visual
---





-------

Make a presentation of you doing the follows. Using Zoom to record a video of the presentation and the link of the video or the video to Canvas. 

[Video Instruction](https://bryant.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=a2216a45-e2c9-4c4a-af02-b0a9015367eb)

-------

_**What to do:**_  Train an ensemble model (Random forest, Adaboost or Gradient Boosting) to recognize three handwritten letters or numbers or objects. Present your model and record the presentation using Zoom or any recording software.  

In the presentation, 

- State clearly what letters/digits/objects you want to predict.

- Show your training and testing data (images).

- Show some efforts to fix any wrong predictions (By adding images that are wrongly classified to the training data and retest with similar images). 

- Discuss issues you run into, if any, and how you fix them. 

- Are you happy with the predictions?

Submit the video or the link of the video to Canvas.


_**Notebooks**_:  

- [Random Forest](https://bryantstats.github.io/math460/python/digits/fa23/image_recognition_rf.html), [Notebook](https://bryantstats.github.io/math460/python/digits/fa23/image_recognition_rf.ipynb)

- [Adaboost](https://bryantstats.github.io/math460/python/digits/fa23/image_recognition_ada.html), [Notebook](https://bryantstats.github.io/math460/python/digits/fa23/image_recognition_ada.ipynb)

- [Gradient Boosting](https://bryantstats.github.io/math460/python/digits/fa23/image_recognition_gb.html), [Notebook](https://bryantstats.github.io/math460/python/digits/fa23/image_recognition_gb.ipynb)




_**Note**_: To improve the accuracy of the model:

- Make the letter/digits dark and thick. 

- Add more images to the training data.

- Add images that are wrongly classified to the training data.

- Let the model see the images more clearly by increasing resolution of the images. This means increasing the value for the `dim` variable.

- Try different values for the tuning parameters

