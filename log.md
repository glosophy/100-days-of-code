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

### Day 32: February 01, 2021
**Today's Progress**: I improved #6: Write a function that finds a subarray whose sum is equal to a given value.

**Thoughts:** I am still trying to figure out if I can solve this one with just one for loop. I can't come up with
a solution that would fit that requirement. I guess I will move onto the next problem and come back to this one later.

**Link to work:** [shortProgProblems.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/shortProgProblems.py)

### Day 33: February 02, 2021
**Today's Progress**: I finally finished putting together [my website](www.glosophy.org)!

**Thoughts:** It took me a good week to finally finish it. This will help me build my portfolio and put my work out there.
I will be updating it regularly with my Medium articles, talks on dataviz, code snippets, and other stuff I will be 
doing in the upcoming months.

**Link to work:** [gLosophy](www.glosophy.org)

### Day 34: February 03, 2021
**Today's Progress**: I started solving #8: Write a function that, given two arrays, finds the longest common subarray present in both of them.
But I could not finish it. 

**Thoughts:** I wrote some sort of pseudo code that helped me get started. I think I'll start doing that from now on 
to make sure I understand the problem first and how to actually iterate through the different elements. I am having 
issues trying to see if `placeholder_list`, which is a chunk of `list1`, is in `list2`. Should I do that with `intersect()`
or what? I might look into that tomorrow.  

**Link to work:** [shortProgProblems.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/shortProgProblems.py)

### Day 35: February 04, 2021
**Today's Progress**: I finally figured out how to solve #8.

**Thoughts:** It is not the most efficient code out there, but after giving this problem more thought, I finally
got the results I wanted! One thing that helped a lot was to write down in a piece of paper some sort of pseudo code, and
simulate a for loop. That helped me a lot to figure out how to iterate through each list.

**Link to work:** [shortProgProblems.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/shortProgProblems.py)

### Day 36: February 05, 2021
**Today's Progress**: I am still trying to find a more efficient way of solving #8 after Santiago [recommended me](https://twitter.com/svpino/status/1357902227554377731?s=20)
to look for a more dynamic solution. 

**Thoughts:** After looking into other people's approaches to longest common substring, I am still hesitant about the approach
I'm using. I would eventually like my function to return the longest common subarray in the form of an array, not
just the length of the array. The latter I have already done, but it's the former that I cannot solve still.

**Link to work:** [shortProgProblems.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/shortProgProblems.py)

### Day 37: February 06, 2021
**Today's Progress**: I am still trying to find a more efficient way of solving #8 after Santiago [recommended me](https://twitter.com/svpino/status/1357902227554377731?s=20)
to look for a more dynamic solution. 

**Thoughts:** I can't still come up with a more dynamic solution than the one I did on February 04. The common subarray
returned by the function still contains an element that does not belong in the common subarray.

**Link to work:** [shortProgProblems.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/shortProgProblems.py)

### Day 38: February 07, 2021
**Today's Progress**: I ended up settling for returning just the length of the common subarray. I followed [this article](https://www.geeksforgeeks.org/longest-common-substring-dp-29/).

**Thoughts:** This dynamic solution worked definitely better than the first approach I took. The first time I used three
for loops to iterate over the elements of both lists. Now I'm only using two for loops, which makes the whole function
less computationally expensive.

**Link to work:** [shortProgProblems.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/shortProgProblems.py)

### Day 39: February 08, 2021
**Today's Progress**: I took a break from Santiago's programming problems to fresh up on [Monte Carlo simulations](https://pbpython.com/monte-carlo.html).
I used [this Interview Query problem](https://www.interviewquery.com/questions/nightly-job) to run a Monte Carlo experiment
and calculate how much the overlapping of two computing jobs would cost to a certain company.

**Thoughts:** Monte Carlo it just a fancy way of saying that one will run an experiment several times to be able to rely on more
accurate probabilities. For this particular problem, I ran the experiment `range(1, 50000, 100)` times. I also plotted
the probabilities aby the number of simulations and the graph shows how the probability converges after 1,000 simulations.

**Link to work:** [MonteCarlo.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Interview%20Questions/MonteCarlo.py)

### Day 40: February 09, 2021
**Today's Progress**: I have been interested in crypto for a long time now, so I decided to start creating some sort of 
a program that tells you when is a good time to sell or buy crypto based on historical prices. I followed [this tutorial](https://medium.com/better-programming/get-the-price-of-cryptocurrencies-in-real-time-using-python-cdaf07516479)
to get a better sense of how to do the web scraping with `bs4`. 

**Thoughts:** I will try to make the program send me texts everytime there is a price update. I could probably create 
two different types of notifications: regular notifications and emergency notifications. Regular notifications would be sent 
probably every two hours, and emergency notifications would be sent when the crypto price moves above/below a certain
threshold.

**Link to work:** [Crypto.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/Crypto.py)

### Day 41: February 10, 2021
**Today's Progress**: I'm still working on trying to make this app send me text messages or Telegram notifications when
the crypto prices either increase or decrease. 

**Thoughts:** I am wondering if this program should eventually turn into a mobile app or web app for personal use only.
Some questions that I still need answer for: how can I have an app running all day? I guess I have to host it somewhere.
If so, where? How much would it cost? I know I'm some steps ahead, but it would be good to start looking into this early
rather than late.

**Link to work:** [Crypto.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/Crypto.py)

### Day 42: February 11, 2021
**Today's Progress**: Making progress with the notifications sent to my phone. That's what I have been working on. 

**Thoughts:** I am still wondering whether I should follow [this tutorial](https://realpython.com/python-bitcoin-ifttt/) 
for the Telegram messages.

**Link to work:** [Crypto.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/Crypto.py)

### Day 43: February 12, 2021
**Today's Progress**: Making progress with the notifications sent to my phone. 

**Thoughts:** I am still wondering whether I should follow [this tutorial](https://realpython.com/python-bitcoin-ifttt/) 
for the Telegram messages.

**Link to work:** [Crypto.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/Crypto.py)

### Day 44: February 13, 2021
**Today's Progress**: I am looking for sources of bots that automatically trade crypto for you based on some threshold
or price limit. I would like this bot to trade crypto for me on KuCoin.  

**Thoughts:** Is that doable? How do I connect my wallet to this bot and tell it to trade for me? How do I signal the bot
when to trade (buy/sell)? Will look into that tomorrow.

**Link to work:** [Crypto.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/Crypto.py)

### Day 45: February 14, 2021
**Today's Progress**: Happy Valentine's Day! I took a break from the crypto bot and kept on solving [Santiago's programming challenge](https://twitter.com/svpino/status/1354048200601198593?s=20).
I tackled problem #9: Write a function that finds the length of the shortest array that contains both input arrays as
subarrays.   

**Thoughts:** This problem was a difficult one, but having already worked on #8 gave me a pretty good intuition on
how to approach this problem. I iterated through the two lists to find the length of the shortest common array present
in the two arrays passed as inputs.

**Link to work:** [shortProgProblems.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/shortProgProblems.py)

### Day 46: February 15, 2021
**Today's Progress**: I kept on working on [Santiago's short programming problems](https://twitter.com/svpino/status/1354048200601198593?s=20).
They are so much fun and made me learned a lot. Not quite sure how to tackle #10 yet: Write a function that, given an 
array, determines if you can partition it in two separate subarrays such that the sum of elements in both subarrays 
is the same. 

**Thoughts:** I'm afraid I'll have to loop through the whole array, and find the nth element that splits it into
two arrays whose sum is the same. I got stuck after finding the splitting point.

**Link to work:** [shortProgProblems.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/shortProgProblems.py)

### Day 47: February 16, 2021
**Today's Progress**: I finally figured out how to solve #10! I want to say it was easier than I expected. As you move onto 
the last programing problems they tend to get harder and harder. I do not know why I thought this was going to be way
hasrder than what it ended up being. Hopefully, I'll get an answer fron Santiago and see if I did a good job.

**Thoughts:** Sometimes when I get stuck solving a problem I tend to get overwhelmed and think that I will never be able
to solve it. But usually if I give it at least a day, I can even come up with a better and more efficient solution. I 
need to start being better at not getting frustrated very rapidly. 

**Link to work:** [shortProgProblems.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/shortProgProblems.py)

### Day 48: February 17, 2021
**Today's Progress**: I think I finished all of [Santiago's short programming problems](https://twitter.com/svpino/status/1354048200601198593?s=20)!
I just finished #11. Write a function that, given an array, divides it into two subarrays, such as the absolute 
difference between their sums is minimum.

**Thoughts:** I am sure there's still a way of reducing the lines of code by almost half. But I think I just like writing
down every single step and have a better understanding of what is going on in the middle of the function. In any case,
I will give it another try and try to find a way of writing less lines of code. 

**Link to work:** [shortProgProblems.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/shortProgProblems.py)

### Day 49: February 18, 2021
**Today's Progress**: I managed to write #11 using fewer lines of code.

**Thoughts:** I am still not sure about the benefits of wirting code using fewr lines of code. Unless I'm not adding
more for loops the code does not get computationally more expensive. I think there's still benefits in including all steps 
when building a function. Specially if someone else is going to end up eventually using the code. There are benefits on both
sides, of course, but I'd rather make it more specific than less easy to understand.

**Link to work:** [shortProgProblems.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/shortProgProblems.py)

### Day 50: February 19, 2021
**Today's Progress**: I started doing some interesting "artsy" stuff with matplotlib.

**Thoughts:** I am sruggling a little bit to find another project to work on. Specially now that I've finished the
programming problems that were taking much of my `#100DaysOfCode` time. I'll probably start solving some programming questions
from interview websites.

**Link to work:** [matplotlib.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/matplotlib.py)

### Day 51: February 20, 2021
**Today's Progress**: I continued drawing shapes with `matplotlib`.

**Thoughts:** I think playing around with `matplotlib` is something I really enjoy. I will be making more shapes or artsy
charts. This is something I really enjoyed!

**Link to work:** [matplotlib2.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/matplotlib2.py)

### Day 52: February 21, 2021
**Today's Progress**: Brushing up on my Javascript skills.

**Thoughts:** I am working on creating an interactive game that takes you through the whole path of getting a green card.

**Link to work:** [prototypeGame.js](https://github.com/glosophy/100-days-of-code/blob/main/Projects/ImmigrationGame/prototypeGame.js)

### Day 53: February 22, 2021
**Today's Progress**: I have created a folder with interview questions I will start solving from now on until I come up
with a nice project to work on for the rest of this challenge or at least for the next 20 days. 

**Thoughts:** Now that I have finished [Santiago's short programming problems](https://twitter.com/svpino/status/1354048200601198593?s=20),
 it has become really hard for me to find something to work on. I wonder if this is something that usually happens to 
 people who start this challenge. 

**Link to work:** [Interview Questions](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions)

### Day 54: February 23, 2021
**Today's Progress**: Michael Corleone's "Just when I thought I was out, they pull me back in" is exactly what happened
with [Santiago's new programming problems](https://twitter.com/svpino/status/1364448470413828097?s=20). I'm back solving
these problems. Will move on to interview questions/problems after I solve these. 

**Thoughts:** The first one was kind of simple. I had to build a function that given two strings "s1" and "s2", creates 
a new string by appending "s2" in the middle of "s1". Although it was pretty straight forward, it was good working with
strings.

**Link to work:** [ProgrammingChallenge.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/ProgrammingChallenge.py)

### Day 55: February 24, 2021
**Today's Progress**: I kept solving Santiago's programming challenge. This time I built a function that, given a tring, 
reorders its characters so all the uppercase letters come first, and the lowercase letter come last.

**Thoughts:** Again, this one was prety easy to tackle. I learned about `isalnum()` and, again, got to work with strings,
which I don't do often.

**Link to work:** [ProgrammingChallenge.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/ProgrammingChallenge.py)

## Day 56: February 25, 2021
**Today's Progress**: Today was a quick session, but I got to work with dictionaries, which I don't do often. I solved
problem #3: Build a function that, given a list of values, returns a dictionary with the number of occurrences of each
value in the list.

**Thoughts:** This problem reminded me I should probably work with dictionaries more often. I should maybe do a full
coding session related to dictionaries...?

**Link to work:** [ProgrammingChallenge.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/ProgrammingChallenge.py)

## Day 57: February 26, 2021
**Today's Progress**: Today was a quick session, but I got to work with dictionaries, which I don't do often. I solved
problem #3: Build a function that, given a list of values, returns a dictionary with the number of occurrences of each
value in the list.

**Thoughts:** This problem reminded me I should probably work with dictionaries more often. I should maybe do a full
coding session related to dictionaries...?

**Link to work:** [ProgrammingChallenge.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/ProgrammingChallenge.py)

## Day 58: February 27, 2021
**Today's Progress**: I finished [Santiago's programming problems](https://twitter.com/svpino/status/1364448470413828097?s=20)!
I can't emphasize enough how much I enjoyed this challenge. 

**Thoughts:** I think this type of problems are helpful to keep your feet on the ground. They let you go back to the
basics and, for a moment, remove yourself from the more complex algorithms to just work on the ABC of programming.

**Link to work:** [ProgrammingChallenge.py](https://github.com/glosophy/100-days-of-code/blob/main/Projects/Random/ProgrammingChallenge.py)

## Day 59: February 28, 2021
**Today's Progress**: I started working on the Machine Learning Theory module in [Confetti AI](https://www.confetti.ai/curriculum). 
I built functions that calculate means, medians, and specific percentiles of a given list of numbers. I also answered
questions on bias, probability, distributions, ROC curve, and AUC.

**Thoughts:** For the question about percentiles, the proposed solution by Confetti AI used the `math` module of Python
to calculate `math.ceil` and `math.floor`. What I did instead, was to find a way to avoid using `math`. I'm finding all 
these questions and coding problems very useful to practice for coding interviews in general and to brush up on concepts 
I had somewhat forgotten or that I need to have more present.

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 60: March 1, 2021
**Today's Progress:** I continued answering questions from the Machine Learning Theory module in [Confetti AI](https://www.confetti.ai/curriculum). 
I built two functions to add matrices and another one that claculates the Euclidean distance. 

**Thoughts:** It was good to review some concepts related to normal distribution, probability, and linear algebra. The 
questions definitely get harder and harder as you make progress. 

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 61: March 2, 2021
**Today's Progress:** I have finished all the tasks from the Machine Learning Theory module in [Confetti AI](https://www.confetti.ai/curriculum). 
I built a function to calculate the Manhattan distance between a set of points. I also started a new module: Core Machine Learning.
This one covers algorithms and techniques that fall into standard (non-deep) machine learning.

**Thoughts:** I learned a lot with this first module on Machine Learning Theory. I think what I enjoyed the most were
the programming problems and finding different solutions to them. Overall, I liked it. 

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 62: March 3, 2021
**Today's Progress:** Wow! I started [Confetti AI's Core Machine Learning module](https://www.confetti.ai/curriculum). 
I am learning a lot! Specially with the multiple-choice questions.

**Thoughts:** I created a function to fit some (X, Y) tuples into a linear function. I remembered my time series class,
and fortunately could come up with a solution to that problem.  

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 63: March 4, 2021
**Today's Progress:** I continued answering theory questions and solving the programming problems in
 [Confetti AI's Core Machine Learning module](https://www.confetti.ai/curriculum). I can't emphasize enough how much I'm learning.

**Thoughts:** The programming problems are getting harder and harder. Sometimes I had to look at the solution to see
at least how to approach the problem.

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 64: March 5, 2021
**Today's Progress:** I continued answering theory questions and solving the programming problems in
 [Confetti AI's Core Machine Learning module](https://www.confetti.ai/curriculum). It was only questions I answered, but
 got to work on some very difficult programming problems.  

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 65: March 6, 2021
**Today's Progress:** I'm still solving problems from [Confetti AI's Core Machine Learning module](https://www.confetti.ai/curriculum). 
Today I really struggled with the linear interpolation programming problem. 

**Thoughts:** This particular problem asked to fit a line using gradient descent. Although I have done that before with
2x1 inputs for a neural network, I had never seen it with linear regression. 

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 66: March 7, 2021
**Today's Progress:** I finally finished [Confetti AI's Core Machine Learning module](https://www.confetti.ai/curriculum)!
Now I'm moving onto their Metrics module. 

**Thoughts:** The remaining coding exercises in the section were really hard to implement. I had to do a lot of Google
searches and sometimes look at their solution, which is not the best, IMO, but at least it helped me realize how to
approach these types of problems. I also realized I need to work on `class` in Python. I kinda forgot how to create and
manipulate them.

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 67: March 8, 2021
**Today's Progress:** Woohoo! I finished [Confetti AI's Metrics module](https://www.confetti.ai/curriculum)!
I got to answer questions on label imabalances, dealing with mising data, least squares, correlations, etc. 

**Thoughts:** The programming exercise was focused on calculating the F1 score given two arrays. I iterated through
the list of true and predicted labels and calculated the true postives, false negatives, and false positives.

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 68: March 9, 2021
**Today's Progress:** I started [Confetti AI's Deep Learning module](https://www.confetti.ai/curriculum)!
 
**Thoughts:** Strangely so, I am doing better on this module than I did on the previous ones.

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 69: March 10, 2021
**Today's Progress:** I just finished [Confetti AI's Deep Learning module](https://www.confetti.ai/curriculum)!
I answered all the questions about SGD, vanishing gradients, hidden layers, etc. 

**Thoughts:** I definitely loved this module. It helped me brush up on some deep learning concepts I have not used
in a while. 

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 70: March 11, 2021
**Today's Progress:** I jumped straight to the Numpy module in [Confetti AI's Engineering and Tools section](https://www.confetti.ai/curriculum)!
I built functions to calculate cosine similarity, euclidean distance, the softmax scores of the vector, etc.

**Thoughts:** This was so much fun! I always love going back to some linear algebra concepts and refresh them.

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 71: March 12, 2021
**Today's Progress:** I keep enjoying [Confetti AI's Engineering and Tools section](https://www.confetti.ai/curriculum)!
I normalized an array (subtract the mean and divide by the standard deviation), and calculated the average of a matrix.

**Thoughts:** This keep getting more fun. 

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 72: March 13, 2021
**Today's Progress:** I am still coding along [Confetti AI's Engineering and Tools section](https://www.confetti.ai/curriculum)!
I created functions that compare two matrices and tells you whether they are the same or not, another that takes two values
as inputs and returns an array consisting of all integer values between start and end.

**Thoughts:** Although some of these funtions I have to build are easy enough to do think through, I am still learning
and keep remembering some basic concepts. 

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 73: March 14, 2021
**Today's Progress:** I am still coding along [Confetti AI's Engineering and Tools section](https://www.confetti.ai/curriculum)!
I created functions that calculates given percentiles for given arrays, Manhattan distance, and another one that 
multiplies matrices.

**Thoughts:** Many of the functions I built today I had already done in previous exercises, with minor changes. But I 
still took the time to build them from scratch and go through the whole thinking process.  

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 74: March 15, 2021
**Today's Progress:** I just finished the numpy module from [Confetti AI's Engineering and Tools section](https://www.confetti.ai/curriculum)!
The functions I created compare two vectors to find common entries between them, another one normalizes a matrix by
subtracting the columns mean from the matrix entry, and another one that scales a vector given a certain number.

**Thoughts:** Apart from the implementation of each function, I learned the why behind each of these. For example, subtracting
the columns mean from each of the matrix entries has the effect of zero-centering the data and is often done in 
algorithms such as PCA or when running computer vision models. Another thing I learned is that 
checking for `NaN` values is a common thing when building deep learning models, as instable training procedures can generate NaNs in the weights.  

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 75: March 16, 2021
**Today's Progress:** I started the Data Manipulation, `pandas`, and Spark module from [Confetti AI's Engineering and Tools section](https://www.confetti.ai/curriculum)!
I am going over basic functions that compute the median and standard deviation of certain columns, or that retrieve the 
top 3 entries of the dataframe.  

**Thoughts:** I am excited about the next set of coding questions that cover `BeautifulSoup`! It never hurts to go over
the basic concepts of slicing dataframes and retrieve certain values from a column given a condition.

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 76: March 17, 2021
**Today's Progress:** I am coding along the Data Manipulation, `pandas`, and Spark module from [Confetti AI's Engineering and Tools section](https://www.confetti.ai/curriculum)!
I used `BeautifulSoup` to scrape some table values from a Wikipedia page.

**Thoughts:** This coding problem took me longer than I thought it would. Given the structure of the code, I had to stick
to how the code was presented and build on top of that. I would have done things different, but I am glad I was able
to figure it out. I wrote [an article in Spanish](https://gsutter.medium.com/c%C3%B3mo-leer-tablas-html-con-pandas-ef1c59ffa81a) 
about pulling data from tables on Wikipedia. I find this way of doing this easier and there is no need to use `BeautifulSoup`.

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 77: March 18, 2021
**Today's Progress:** I am still coding along the Data Manipulation, `pandas`, and Spark module from [Confetti AI's Engineering and Tools section](https://www.confetti.ai/curriculum)!
I am now using `pandas` to manipulate different dataframes.

**Thoughts:** Although I am really enjoying the coding problems, I still find it somewhat difficult to adapt to how the
function is presented. Again, I would probably start tackling the problem in a different way. But I like how it makes
me think about or approach the problem.

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 78: March 19, 2021
**Today's Progress:** I am still coding along the Data Manipulation, `pandas`, and Spark module from [Confetti AI's Engineering and Tools section](https://www.confetti.ai/curriculum)!
I am now using `pandas` to manipulate different dataframes.

**Thoughts:** Although I am really enjoying the coding problems, I still find it somewhat difficult to adapt to how the
function is presented. Again, I would probably start tackling the problem in a different way. But I like how it makes
me think about or approach the problem.

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 79: March 20, 2021
**Today's Progress:** I am still coding along the Data Manipulation, `pandas`, and Spark module from [Confetti AI's Engineering and Tools section](https://www.confetti.ai/curriculum)!
I had to build a function that calculates the number of missing days in a given dataset.

**Thoughts:** I rarely work with dates in dataframes, so this function was interesting to build.

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 80: March 21, 2021
**Today's Progress:** I am almost done with the programming exercises from the Data Manipulation, `pandas`, and Spark module from [Confetti AI's Engineering and Tools section](https://www.confetti.ai/curriculum)!
I built functions where I had to handle strings in a `pandas` dataframe. It was really interesting and fun working with 
and iterating through strings in a dataframe.

**Thoughts:** These exercises reminded me of one of the [first projects](https://medium.com/analytics-vidhya/https-medium-com-gss-yaim-yet-another-imessage-mining-1bb0d812b002) I worked on where I analyzed the text messages 
between a friend of mine and I.

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 81: March 22, 2021
**Today's Progress:** I am almost done with the programming exercises from the Data Manipulation, `pandas`, and Spark module from [Confetti AI's Engineering and Tools section](https://www.confetti.ai/curriculum)!
I built functions where I had to handle strings in a `pandas` dataframe. I worked with the baby names dataset, and the 
spam emails.

**Thoughts:** I liked these exercises a lot!

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 82: March 23, 2021
**Today's Progress:** I kept working on the same module from Confetti AI. I built five functions where I, again, had to 
manipulate strings within a datframe. 

**Thoughts:** These exercises were somewhat easy to code, but still a tad challenging.

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)

# Day 83: March 24, 2021
**Today's Progress:** I just finished working on the Data Manipulation, `pandas`, and Spark module from [Confetti AI's Engineering and Tools section](https://www.confetti.ai/curriculum)!

**Thoughts:** I'm glad I finished this module! It helped me brush up on my `pandas` skill.

**Link to work:** [Confetti AI](https://github.com/glosophy/100-days-of-code/tree/main/Projects/Interview%20Questions/ConfettiAI)
