#!/usr/bin/env python
# coding: utf-8

# # Course project guidelines
# 
# Your assignment for the course project is to formulate and answer a question of your choosing based on one of the following datasets:
# 
# 1. ClimateWatch historical emissions data: greenhouse gas emissions by U.S. state 1990-present
# 2. World Happiness Report 2023: indices related to happiness and wellbeing by country 2008-present
# 3. Any dataset from the class assignments or mini projects
# 
# A good question is one that you want to answer. It should be a question with contextual meaning, not a purely technical matter. It should be clear enough to answer, but not so specific or narrow that your analysis is a single line of code. It should require you to do some nontrivial exploratory analysis, descriptive analysis, and possibly some statistical modeling. You aren't required to use any specific methods, but it should take a bit of work to answer the question. There may be multiple answers or approaches to contrast based on different ways of interpreting the question or different ways of analyzing the data. If your question is answerable in under 15 minutes, or your answer only takes a few sentences to explain, the question probably isn't nuanced enough.
# 
# ## Deliverable
# 
# Prepare and submit a jupyter notebook that summarizes your work. Your notebook should contain the following sections/contents:
# 
# * **Data description**: write up a short summary of the dataset you chose to work with following the conventions introduced in previous assignments. Cover the sampling if applicable and data semantics, but focus on providing high-level context and not technical details; don't report preprocessing steps or describe tabular layouts, etc.
# * **Question of interest**: motivate and formulate your question; explain what a satisfactory answer might look like.
# * **Data analysis**: provide a walkthrough with commentary of the steps you took to investigate and answer the question. This section can and should include code cells and text cells, but you should try to focus on presenting the analysis clearly by organizing cells according to the high-level steps in your analysis so that it is easy to skim. For example, if you fit a regression model, include formulating the explanatory variable matrix and response, fitting the model, extracting coefficients, and perhaps even visualization all in one cell; don't separate these into 5-6 substeps.
# * **Summary of findings**: answer your question by interpreting the results of your analysis, referring back as appropriate. This can be a short paragraph or a bulleted list.
# 
# 
# ## Evaluation
# 
# Your work will be evaluated on the following criteria:
# 
# 1. Thoughtfulness: does your question reflect some thoughtful consideration of the dataset and its nuances, or is it more superficial?
# 2. Thoroughness: is your analysis an end-to-end exploration, or are there a lot of loose ends or unexplained choices?
# 3. Mistakes or oversights: is your work free from obvious errors or omissions, or are there mistakes and things you've overlooked?
# 4. Clarity of write-up: is your report well-organized with commented codes and clear writing, or does it require substantial effort to follow?

# In[11]:


import numpy as np
import pandas as pd
import altair as alt
from sklearn.decomposition import PCA
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
# disable row limit for plotting
alt.data_transformers.disable_max_rows()
# uncomment to ensure graphics display with pdf export
alt.renderers.enable('mimetype')
alt.renderers.enable('default')


# In[12]:


wh_raw = pd.read_csv('data/whr-2023.csv')
wh_raw.head()


# In[15]:


fig1 = alt.Chart(wh_raw).mark_bar(opacity = 0.1).encode(
    y = 'Gap',
    x = alt.X('Measure', scale = alt.Scale(zero = False), title = ''),
    color = 'Gap type'
).properties(
    width = 100,
    height = 100
).facet(
    column = alt.Column('Socioeconomic variable')
).resolve_scale(x = 'independent')


# In[ ]:


# edit



#KDE part
Question: 
How does 'Life Ladder' correlate with 'Log GDP per capita'?
What impact does 'Healthy life expectancy at birth' seem to have on this relationship?
Are there any noticeable trends or outliers?

Ans:
#Based on the scatter plot for 2022, a positive correlation between 'Log GDP per capita' and 'Life Ladder' scores is likely, indicating that higher economic status is associated with greater life satisfaction. The color variation, representing 'Healthy life expectancy at birth', suggests a connection between health and well-being, with better health outcomes often coinciding with higher GDP and life satisfaction. The spread and density of the data points reveal variations in economic and life satisfaction profiles across different countries, providing insight into the complex interplay between economic, health, and well-being factors.

whr_2022 = wh_raw[wh_raw['year'] == 2022]

fig2, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=whr_2022, x='Log GDP per capita', y='Life Ladder', hue='Healthy life expectancy at birth', palette='viridis', s=100, ax=ax)
ax.set_title('Relationship between GDP and Life Satisfaction in 2022')
ax.set_xlabel('Log GDP per Capita')
ax.set_ylabel('Life Ladder')

fig2


Question: 
Which "Life Ladder" scores are most common across all countries?
Which "Life Ladder" scores are rare?
How spread out are the "Life Ladder" scores?
Are the scores spread evenly or irregularly across the dataset?

Ans:
The histogram shows a range of life satisfaction scores across various countries, with the KDE providing a smooth representation of this distribution. The  'Life Ladder' scores would be indicated from 1 to 8, showing where scores are concentrated. The spread of the 'Life Ladder' scores from the width of the histogram and the KDE curve, tells  single-peaked distribution would indicate more uniformity in life satisfaction levels.

data = wh_raw['Life Ladder']

fig3, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data, bins=np.arange(data.min(), data.max(), 0.03), stat='density', kde=False, color='blue', alpha=0.6, label='Histogram', ax=ax)

density = gaussian_kde(data)
xs = np.linspace(data.min(), data.max(), 300)
density.covariance_factor = lambda : .25
density._compute_covariance()

ax.plot(xs, density(xs), color='black', label='KDE')

ax.set_title('Density Scale Histogram with KDE for Life Ladder Scores')
ax.set_xlabel('Life Ladder Score')
ax.set_ylabel('Density')
ax.legend()

fig3


fig4, ax = plt.subplots(figsize=(12, 8))
sns.histplot(wh_raw['Life Ladder'], bins=30, stat='density', kde=False, color='skyblue', alpha=0.6, label='Histogram', ax=ax)

kde_density = gaussian_kde(wh_raw['Life Ladder'])
kde_xs = np.linspace(wh_raw['Life Ladder'].min(), wh_raw['Life Ladder'].max(), 300)
kde_density.covariance_factor = lambda : .25
kde_density._compute_covariance()

ax.plot(kde_xs, kde_density(kde_xs), color='darkblue', label='KDE')

ax.set_title('Global Distribution of Life Ladder Scores Over Time')
ax.set_xlabel('Life Ladder Score')
ax.set_ylabel('Density')
ax.legend()

fig4

Question:
Conditional Distributions of Life Satisfaction Scores Relative to GDP

Ans:
The plot has a considerable impact on the distribution of life satisfaction scores across countries, with wealthier countries tending to have higher overall life satisfaction.


median_gdp = wh_raw['Log GDP per capita'].median()

wh_raw['above_median_gdp'] = wh_raw['Log GDP per capita'] > median_gdp

fig5, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(data=wh_raw, x='Life Ladder', hue='above_median_gdp', shade=True, common_norm=False, palette=['blue', 'orange'], alpha=0.5, ax=ax)
ax.set_title('Distribution of Life Satisfaction Scores Based on GDP per Capita')
ax.set_xlabel('Life Ladder Score')
ax.set_ylabel('Density')
ax.legend(title='Above Median GDP', labels=['No', 'Yes'])

fig_gdp = plt.gcf()

fig5


Question:
Multi-panel Visualization of Life Satisfaction and GDP with Health Context

Ans:
The jointplot of 'Log GDP per capita' versus 'Life Ladder' scores, with the marginal density plots and points colored by health expectancy, suggests a nuanced relationship between economic status, health, and life satisfaction. Countries with higher GDP per capita generally appear to have higher life satisfaction scores, and this trend is more pronounced in nations with above-median healthy life expectancy. The marginal density plots reveal the distribution of GDP and life satisfaction scores separately, further emphasizing the positive correlation between economic prosperity and well-being.


jp = sns.jointplot(data=wh_raw, x='Log GDP per capita', y='Life Ladder', hue='high_life_expectancy', kind="scatter")

jp.fig.suptitle("GDP vs Life Satisfaction with Health Context", fontsize=16)
jp.set_axis_labels('Log GDP per Capita', 'Life Ladder Score', fontsize=12)
jp.fig.tight_layout()
jp.fig.subplots_adjust(top=0.95) # Adjust the title position

plt.show()

