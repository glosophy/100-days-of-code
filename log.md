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
past week. The main idea is to see how all these methods perform on the US Census dataset. I found [this article](https://www.pluralsight.com/guides/cleaning-up-data-from-outliers)
about cleaning up data from outliers very useful.

**Link to work:** [emsembleMethods.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/XGBoost/emsembleMethods.py)

### Day 16: January 16, 2021
**Today's Progress**: I finished preparing the different models and structured the project so it returns a dataframe
with the models and its evaluation metrics. I mapped the categorical features to a dictionary that contained its corresponding
numerical values. To do that, I followed [this DataCamp tutorial](https://www.datacamp.com/community/tutorials/categorical-data).

**Thoughts:** The XGBoost model is overperforming the rest. Second is random forest. I might tune each of them to try to
get the best performances from each of them.

**Link to work:** [emsembleMethods.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/XGBoost/emsembleMethods.py)

### Day 17: January 17, 2021
**Today's Progress**: I kept working on the different models and tuning them. A final dataframe summarizes the different accuracies.

**Thoughts:** The XGBoost model outperforms the rest, followed by Gradient Boosting and Bagging.

**Link to work:** [emsembleMethods.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/XGBoost/emsembleMethods.py)

### Day 18: January 18, 2021
**Today's Progress**: I worked on the models' performance more by fine-tuning the models. Overall, it was worth allocating
more time on this task since it increased their overall performance.

**Thoughts:** After doing some `GridSearchCV`, I managed to get R2 scores of around 50% for all models except for AdaBoost. 
All RMSEs decreased. AdaBoost was the one with the highest RMSE. Of all six models, the over-performing one was XGBoost,
with an R2 score of 57% (the features explain 57% of the change in the income wage) and a RMSE of 26,128.
The second best model was the GradBoost, with an R2 score of 55% and an RMSE of 26,738.

**Link to work:** [emsembleMethods.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/XGBoost/emsembleMethods.py)

### Day 19: January 19, 2021
**Today's Progress**: I cleaned and did some EDA on a direct phone call marketing campaign by a Portuguese bank. The dataset
was downloaded from Kaggle and can be found [here](https://www.kaggle.com/yufengsui/portuguese-bank-marketing-data-set). I also
used [this Stackoverflow link](https://stackoverflow.com/questions/38088652/pandas-convert-categories-to-numbers) to encode categorical
columns in pandas. 

**Thoughts:** `seaborn` > `matplotlib`. 

**Link to work:** [binaryClassification.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/Ensemble%20Binary/binaryClassification.py)

### Day 20: January 20, 2021
**Today's Progress**: I ran some binary classification models on the bank marketing campaign dataset and summarized
the results in a dataframe. 

**Thoughts:** When working with imbalanced datasets, training accuracies don't tell much. Actually, they can be misleading.
Instead, one should look at the precision and recall figures and evaluate the model based on those metrics. 
The recall metric is basically the ability of the model to find all the positive samples. The precision number shows
how many examples were classified correctly. If using `sklearn.metrics.recall_score` for binary classification, it
is important to pass `average='binary'` as argument.

**Link to work:** [binaryClassification.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/Ensemble%20Binary/binaryClassification.py)

### Day 21: January 21, 2021
**Today's Progress**: I started working on a very easy SVM model. Will try it on another dataset tomorrow.

**Thoughts:** Although less interpretable, SVM is a very powerful model when it comes to classification problems. It is
 also a very flexible model to work with. In this case, it classified 100% of the classes correctly.

**Link to work:** [SVM.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/SVM/SVM.py)

### Day 22: January 22, 2021
**Today's Progress**: I ran another SVM model on a voice dataset. Will try a non-linear kernel tomorrow.

**Thoughts:** Although less interpretable, SVM is a very powerful model when it comes to classification problems. It is
 also a very flexible model to work with. In this case, it classified 100% of the classes correctly.

**Link to work:** [SVM2.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/SVM/SVM2.py)

### Day 23: January 23, 2021
**Today's Progress**: I prepared and cleaned a dataset from UCI on [chronic kidney disease](https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease).

**Thoughts:** As I was EDAing the dataset I ran into the following question: How do you go around using `StandardScaler()`
when you have both categorical and numeric features measured at different scales? I read [this article](https://towardsai.net/p/data-science/how-when-and-why-should-you-normalize-standardize-rescale-your-data-3f083def38ff)
but it only explained how and when to standardize/normalize your data. It didn't say much about this problem I ran into.
I guess I'll do more research and try to pick up where I left off tomorrow when I have more info.

**Link to work:** [KNN.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/KNN/KNN.py)

### Day 24: January 24, 2021
**Today's Progress**: I realized some of the features in the dataset had missing values, so I filled those `nan` with the mode.
I used `pd.get_dummies` on the categorical features and then `MinMaxScaler()` to coerce all the features
to a 0-1 scale. I evaluated the model on the accuracy score, classification report, and confusion matrix.

**Thoughts:** Normalizing the data made all the difference. The model went from an accuracy of 68% to one of 97.5%.
The classification report threw a weighted precision of 0.69 and weighted recall of 0.69 for the model with no feature
normalization. After applying `MinMaxScaler()`, those figures increased to 0.98 and 0.97, respectively. One thing that
I take from this modeling exercise is that categorical features are dummies, and continuous/nominal features
should be standardized or normalized before fitting them into the model.

**Link to work:** [KNN.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/KNN/KNN.py)

### Day 25: January 25, 2021
**Today's Progress**: Started working on the dataset to get it ready for the model.

**Link to work:** [naiveBayes.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/Naive%20Bayes/naiveBayes.py)

### Day 26: January 26, 2021
**Today's Progress**: I finished cleaning the dataset and ran `GaussianNB()` on the dataset.

**Thoughts:** I am not sure about how I feel about `GaussianNB()`. I would have thought this model was going to perform 
better than it actually did. I got an accuracy of 79% with a recall of 0.32 for the `1` class and 0.81 for the `0` class.
The precision score for both classes was 0.68 and 0.81, respectively. If I had to choose a model for this classification
problem, I would use a different classifier, maybe `XGBoost`.

**Link to work:** [naiveBayes.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/Naive%20Bayes/naiveBayes.py)

### Day 27: January 27, 2021
**Today's Progress**: I followed [this article](https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c) 
to perform k-Means on some random blobs I created.

**Thoughts:** I wonder how would `KMeans` work on a particular dataset. 

**Link to work:** [kMeans.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Machine%20Learning/KMeans/kMeans.py)

### Day 28: January 28, 2021
**Today's Progress**: I started a short programming problems challenge by Santiago. I ran into [his tweet](https://twitter.com/svpino/status/1354048200601198593?s=20)
a couple of weeks ago and thought it would be good to do these problems as part of the `#100DaysOfCode`. Today I finished
the first two problems. 

**Thoughts:** The first problem consisted of sorting an array in place. I remembered how to do that from my Data Mining class.
`list[::-1]` is a savior. The second problem took me a little bit more of time. At first I didn't realize I needed two for loops
to properly sort the array before finding the missing value. But once I sorted that out (no pun intended), the rest came easy.
I am really having a lot of fun with these problems. 

**Link to work:** [shortProgProblems.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/shortProgProblems.py)

### Day 29: January 29, 2021
**Today's Progress**: I am still working on [Santiago's programming challenge](https://twitter.com/svpino/status/1354048200601198593?s=20). I solved #3: Write a function that finds 
the duplicate number in an unsorted array containing every number from 1 to 100. But I got a little stuck with #4.

**Thoughts:** The first problem I tackled today was somewhat easy. I also used much of the code I had written yesterday.
I am still not entirely sure about how to approach #4. It looks like when I use `list.remove(element)`, the elements is
removed from the list, but when the for loop starts back again, the length of the list is being overwritten by the 
length of the list that is passed in the function and it doesn't take into account the removed element in the inner loop. 

**Link to work:** [shortProgProblems.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/shortProgProblems.py)

### Day 30: January 30, 2021
**Today's Progress**: I continued doing [Santiago's programming challenge](https://twitter.com/svpino/status/1354048200601198593?s=20).
I solved problems #4, 5, 6, and 7. Got stuck with number 8 somehow. Will give it another try tomorrow.

**Thoughts:** Yesterday I had had some issues while tackling #4: Write a function that removes every duplicate number 
in an unsorted array containing every number from 1 to 100. I don't know how I didn't figure this out yesterday... But I eventually
ended up creating an empty array, and `if is in` the empty array, I appended all numbers in the array passed in the function.

**Link to work:** [shortProgProblems.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/shortProgProblems.py)

### Day 31: January 31, 2021
**Today's Progress**: I edited the code I had written for #5: Write a function that finds the largest and smallest number in an unsorted array.
In [this tweet](https://twitter.com/svpino/status/1356034303810039812?s=20), Santiago suggested to try to do just a single
for loop instead of two. I also made edits in #6. I got rid of the loop that ordered the array. I felt like I was cheating
or taking a shortcut, and maybe defeating the whole purpose of the "unsorted array" part of the question.

**Thoughts:** I am so glad that I got to revisit those problems. It makes much more sense now, and both #5 and #6 are
computationally less expensive since I am only using one for loop instead of two. 

*Update: just tweeted my new solution to Santiago and [he replied](https://twitter.com/svpino/status/1356388426094927872?s=20) 
with the following: "This solution is `O(n)`. In English, it means that the solution is "linear": the runtime 
grows linearly with the size of the list. That's much faster than the 2-loop solution which was `O(n^2)`."*

**Link to work:** [shortProgProblems.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/shortProgProblems.py)
