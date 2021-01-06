import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('selected_countries.csv')

df['hf_quartile'] = df['hf_quartile'].astype(int)

df2018 = df[df['year'] == 2018]

#Source: https://towardsdatascience.com/sorry-but-sns-distplot-just-isnt-good-enough-this-is-though-ef2ddbf28078

# set properties for sns chart
sns.set(font_scale=1.35, style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

g = sns.FacetGrid(df2018, #the dataframe to pull from
                  row="hf_quartile", #define the column for each subplot row to be differentiated by
                  hue="hf_quartile", #define the column for each subplot color to be differentiated by
                  aspect=10, #aspect * height = width
                  height=1.6, #height of each subplot
                  palette=['#00A75D', '#4EBC84','#91CFA9', '#C8E6D1'] #google colors
                 )

#shade: True/False, shade area under curve or not
#alpha: transparency, lw: line width, bw: kernel shape specification
g.map(sns.kdeplot, "hf_score", shade=True, alpha=1, lw=1.5, bw_method=.2)
g.map(sns.kdeplot, "hf_score", lw=4, bw_method=0.2)
g.map(plt.axhline, y=0, lw=8)

def label(x, color, label):
    ax = plt.gca() #get the axes of the current object
    ax.text(0, .2, #location of text
            label, #text label
            fontweight="bold", color=color, size=25, #text attributes
            ha="left", va="center", #alignment specifications
            transform=ax.transAxes) #specify axes of transformation

g.map(label, "hf_score") #the function counts as a plotting object!

#prevent overlapping issues by 'removing' axis face color
g.fig.subplots_adjust(hspace=-.05)

g.set_titles("") #set title to blank
g.set(yticks=[]) #set y ticks to blank
g.despine(bottom=True, left=True) #remove y-axis
g.axes[3,0].set_xlabel('Human Freedom Score')
g.fig.suptitle("Human Freedom Density Plots by Quartiles", size=26, ha='right', fontweight='bold')
g.fig.subplots_adjust(top=.5)

plt.show()