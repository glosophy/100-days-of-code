import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('selected_countries.csv')

df['hf_quartile'] = df['hf_quartile'].astype(int)
df2018 = df[df['year'] == 2018]

#Source: https://towardsdatascience.com/sorry-but-sns-distplot-just-isnt-good-enough-this-is-though-ef2ddbf28078

g = sns.FacetGrid(df2018, #the dataframe to pull from
                  row="hf_quartile", #define the column for each subplot row to be differentiated by
                  hue="hf_quartile", #define the column for each subplot color to be differentiated by
                  aspect=10, #aspect * height = width
                  height=1.5, #height of each subplot
                  palette=['#4285F4','#EA4335','#FBBC05','#34A853'] #google colors
                 )

#shade: True/False, shade area under curve or not
#alpha: transparency, lw: line width, bw: kernel shape specification

g.map(sns.kdeplot, "hf_score", shade=True, alpha=1, lw=1.5, bw_method=0.2)
g.map(sns.kdeplot, "hf_score", lw=4, bw_method=0.2)
g.map(plt.axhline, y=0, lw=4)

def label(x, color, label):
    ax = plt.gca() #get the axes of the current object
    ax.text(0, .2, #location of text
            label, #text label
            fontweight="bold", color=color, size=20, #text attributes
            ha="left", va="center", #alignment specifications
            transform=ax.transAxes) #specify axes of transformation

g.map(label, "hf_score") #the function counts as a plotting object!

#prevent overlapping issues by 'removing' axis face color
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
g.fig.subplots_adjust(hspace= -.1)

g.set_titles("") #set title to blank
g.set(yticks=[]) #set y ticks to blank
g.despine(bottom=True, left=True) #remove 'spines'

plt.show()