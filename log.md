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
