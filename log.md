# 100 Days Of Code - Log

### Day 1: January 1, 2021
**Today's Progress**: Created a basic Dash app that tells you if Chick-Fil-A is open.

**Thoughts:** I really struggled with CSS, specially aligning all the elements in a div.

**Link to work:** [Chick-Fil-A App](https://github.com/glosophy/100-days-of-code/blob/main/Projects/ChickFilA/ChickFilA.py)

### Day 2: January 2, 2021
**Today's Progress**: Finished putting together the CSS for the webapp.

**Thoughts:** I ended up figuring out how to align each element in each of the divs. Not my fav thing, CSS was somewhat painful.

**Link to work:** [Chick-Fil-A App](https://github.com/glosophy/100-days-of-code/blob/main/Projects/ChickFilA/ChickFilA.py)

### Day 3: January 3, 2021
**Today's Progress**: Putting together a `seaborn` kdeplot using the data from the Human Freedom Index 2020.

**Thoughts:** I still need to fix some small details like axes ticks and axis labels.

**Link to work:** [KDE Plot with HFI quatiles](https://github.com/glosophy/100-days-of-code/tree/main/Projects/HFI)

### Day 4: January 4, 2021
**Today's Progress**: Finished putting together a `seaborn` kdeplot using the data from the Human Freedom Index 2020.

**Thoughts:** I learned how `FacetGrid` works. At first it was kind of confusing, but once you start playing around with it, 
you get a sense of what to expect and how to customize each subplot.

**Link to work:** [KDE Plot with HFI quatiles](https://github.com/glosophy/100-days-of-code/tree/main/Projects/HFI)

### Day 5: January 5, 2021
**Today's Progress**: I decided I would make the whole data gathering, cleaning, and analysis for the [Human Freedom Index](https://cato.org/hfi) automatic.

**Thoughts:** I ran into some issues because in order to download many of the datasets you have to sign in. What I'll do next is 
download all the datasets manually, and then start from there. It was worth trying, though.

### Day 6: January 6, 2021
**Today's Progress**: Re-thinking my #100DaysOfCode journey. I'll be focusing on strengthening my machine learning skills
through different exercises that include decision trees, ransom forest, SVM, KNN, Naive Bayes, etc. 

### Day 7: January 7, 2021
**Today's Progress**: I solved one of my machine learning exercises on decision trees. I ran two tree models: gini and entropy.
I calculated the classification metrics and plotted the trees and confusion matrices.

**Thoughts:** This exercise made me refresh some decision tree concepts (impurity measurements and information gain).
I found [this article](https://towardsdatascience.com/gini-index-vs-information-entropy-7a7e4fed3fcb) to be particularly helpful.

**Link to work:** [decisionTree.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/Decision%20Tree/decisionTree.py)

### Day 8: January 8, 2021
**Today's Progress**: I used a for loop to iterate through a series of parameters to fine-tune the decision tree model.
I used `DecisionTreeClassifier()` from `sklearn` and fine-tuned `max_depth` and `min_samples_leaf` for both gini and entropy criterions.
I really recommend [this article](https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3#:~:text=min_samples_leaf%20is%20The%20minimum%20number,the%20base%20of%20the%20tree.) 
that helped me figure out what each of the parameters meant in `DecisionTreeClassifier()`.

**Thoughts:** I learned what the different parameters mean and how changing them affect the accuracy scores.
The entropy criterion had the best accuracies and AUC overall.

**Link to work:** [decisionTree.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/Decision%20Tree/decisionTree.py)

### Day 9: January 9, 2021
**Today's Progress**: I played around with `GridSearchCV` and applied it to my decision tree model. Everything I did yesterday,
can be done with just seven lines of code. I found [this Medium article](https://medium.com/ai-in-plain-english/hyperparameter-tuning-of-decision-tree-classifier-using-gridsearchcv-2a6ebcaffeda) useful.

**Thoughts:** `GridSearchCV` is way faster and computationally less expensive than what I did yesterday. It will not always
be less expensive computationally, though -that will ultimately depend on how many parameters you want to include
in the grid search.

**Link to work:** [GridSearchTree.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/Decision%20Tree/GridSearchTree.py)

### Day 10: January 10, 2021
**Today's Progress**: I used `BaggingClassifier()` on the UCI banknote dataset.

**Thoughts:** I am trying out different ensembling methods on the same dataset to: 1) better understand how they work,
and 2) compare them with each other. 

**Link to work:** [Bagging.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/Bagging/Bagging.py)

### Day 11: January 11, 2021
**Today's Progress**: I used `RandomForestClassifier()` on the UCI banknote dataset.

**Thoughts:** So far, the `RandomForestClassifier()` is the best model amongst all ensemble methods i have tried. 
Both the training accuracy (99.51) and the AUC (99.99) were the best metrics so far. I am going to try out a different dataset
tomorrow with more features to play around with `clf.feature_importances_`.

**Link to work:** [randomForest.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/Random%20Forest/randomForest.py)

### Day 12: January 12, 2021
**Today's Progress**: I used `RandomForestClassifier()` on the titanic dataset and brushed up some 
dataset cleaning methods. I also used `clf.feature_importances_` to select the most important features in the dataframe.

**Thoughts:** `clf.feature_importances_` is a great method for getting rid of features that might not add a lot to the model.

**Link to work:** [randomForest2.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/Random%20Forest/randomForest2.py)

### Day 13: January 13, 2021
**Today's Progress**: I brushed up on `GradientBoostingClassifier()` and `AdaBoostClassifier()`. 

**Thoughts:** These two models performed better compared to the decision tree, bagging, and random forest. I'll try 
tomorrow these same models on a bigger dataset with more features to see how it performs.

**Link to work:** [adaBoost.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/AdaBoost/adaBoost.py)
and [gradBoosting.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/Gradient%20Boosting/gradBoosting.py)

### Day 14: January 14, 2021
**Today's Progress**: I used `XGBClassifier()` on the banknote dataset.

**Thoughts:** This model was the one that performed the best, throwing an accuracy of 100% and a AUC of 1.00.

**Link to work:** [XGBoost.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/XGBoost/XGBoost.py)

### Day 15: January 15, 2021
**Today's Progress**: I started doing some EDA and cleaning the US Census dataset.

**Thoughts:** Everything I did today will help me get the data ready to run all the ensemble methods I covered during the 
past week. The main idea is to see how all these methods perform on the US Census dataset.

**Link to work:** [emsembleMethods.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/XGBoost/emsembleMethods.py)
