#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import statsmodels.api as sm

data = pd.read_csv('charitydata.csv')


# In[ ]:


st.title("Interactive CharityData Exploration")
st.write("Explore patterns in the Charitydata ")


# In[ ]:


st.title("CharityData Dataset")
st.header("Introduction")
st.write("""
    The main objective of this analysis is to understand which independent variables has the strongest statistical relationship with and influence over the target variable of damt -  dollar amount of donation.
    The goal is to determine the top variables and to use them to target marketing efforts in order to raise the dollar donation for the charity in the future. The data contains demographics, income, and other donation information.
""")


# In[ ]:


st.sidebar.header("Filter Options")


# In[ ]:


# Sidebar Filters for Numerical Variables

kids_range = st.sidebar.slider("Kids (count)", int(data['kids'].min()), int(data['kids'].max()), (0, 5))
avhv_range = st.sidebar.slider("Average home value in donor's neighborhood in 1,000 USDs", int(data['avhv'].min()), int(data['avhv'].max()), (51, 710))
incm_range = st.sidebar.slider("Median family income in donor's neighborhood in 1,000 USDs", int(data['incm'].min()), int(data['incm'].max()), (3, 287))
inca_range = st.sidebar.slider("Average family income in donor's neighborhood in 1,000 USDs", int(data['inca'].min()), int(data['inca'].max()), (14, 287))
plow_range = st.sidebar.slider("Percentage categorize as low income in donor's neighborhood", int(data['plow'].min()), int(data['plow'].max()), (0, 87))
npro_range = st.sidebar.slider("Lifetime number of promotions received to date", int(data['npro'].min()), int(data['npro'].max()), (2, 164))
tgif_range = st.sidebar.slider("Dollar amount of lifetime gifts to date", int(data['tgif'].min()), int(data['tgif'].max()), (23, 1974))
lgif_range = st.sidebar.slider("Dollar amount of largest gifts to date", int(data['lgif'].min()), int(data['lgif'].max()), (3, 642))
rgif_range = st.sidebar.slider("Dollar amount of most recent gift", int(data['rgif'].min()), int(data['rgif'].max()), (1, 173))
tdon_range = st.sidebar.slider("Numbers of months since last denotion", int(data['tdon'].min()), int(data['tdon'].max()), (5, 40))
tlag_range = st.sidebar.slider("Numbers of months between first and second gift", int(data['tlag'].min()), int(data['tlag'].max()), (1, 34))
agif_range = st.sidebar.slider("Average dollar amount of gifts to date", int(data['agif'].min()), int(data['agif'].max()), (1, 73))
damt_range = st.sidebar.slider("Dollar amount of donation in 1,000 USDs", int(data['damt'].min()), int(data['damt'].max()), (0, 27))



# In[ ]:


# Sidebar Filters for Categorical Variables
reg1 = st.sidebar.multiselect("Donor belongs to region 1 (0 = no 1=yes)", options=data['reg1'].unique(), default=data['reg1'].unique())
reg2 = st.sidebar.multiselect("Donor belongs to region 2(0 = no 1=yes)", options=data['reg2'].unique(), default=data['reg2'].unique())
reg3 = st.sidebar.multiselect("Donor belongs to region 3(0 = no 1=yes)", options=data['reg3'].unique(), default=data['reg3'].unique())
reg4 = st.sidebar.multiselect("Donor belongs to region 4(0 = no 1=yes)", options=data['reg4'].unique(), default=data['reg4'].unique())
home = st.sidebar.multiselect("Homeowner (1= homeowner, 0=not a homeowner)", options=data['home'].unique(), default=data['home'].unique())
hinc = st.sidebar.multiselect("Household income (1= lowest 7=Highest)", options=data['hinc'].unique(), default=data['hinc'].unique())
genf = st.sidebar.multiselect("Gender (0=Male, 1=Female)", options=data['genf'].unique(), default=data['genf'].unique())
wrat = st.sidebar.multiselect("Wealth rating (9 = highest and 0 = lowest)", options=data['wrat'].unique(), default=data['wrat'].unique())
donr = st.sidebar.multiselect("Donated (0 = no 1=yes)", options=data['donr'].unique(), default=data['donr'].unique())


# In[ ]:


filtered_data = data[
    (data['damt'].between(*damt_range)) &
    (data['kids'].between(*kids_range)) &
    (data['hinc'].isin(hinc)) &
    (data['home'].isin(home)&
    (data['wrat'].isin(wrat)))
]


# In[ ]:


if st.sidebar.checkbox("Show Filtered Data"):
    st.write(filtered_data)


# In[ ]:


st.header("Correlation of Numerical Variables")
st.write("This heatmap shows the correlation of the variables.")

numeric_charity = filtered_data.apply(pd.to_numeric, errors='coerce')
corr_matrix = numeric_charity.corr()
plt.figure(figsize=(16, 16)) 
sns.heatmap(corr_matrix, annot=True)
st.pyplot(plt)


# In[ ]:


st.header("BoxPlot: damt & home")
st.write(" Boxplot that visualizes the distribution of the damt target variable and the home variable ")


plt.figure(figsize=(8, 6))
sns.boxplot(x='home', y='damt', data=filtered_data)

plt.xlabel('home (0 vs 1)')
plt.ylabel('damt')
plt.title('Boxplot of damt by home')
st.pyplot(plt)


# In[ ]:


st.header("Violin Plot: damt & home")
st.write("Violinplot that visualizes the density and underlying distribution of the damt target variable and the home variable ")

plt.clf()
sns.violinplot(x='home', y='damt', data=filtered_data, inner='box', scale='width')
plt.xlabel('home (0 vs 1)')
plt.ylabel('damt')
plt.title('Violin Plot of damt by home')
st.pyplot(plt)


# In[ ]:


st.header("BoxPlot: damt & wrat")
st.write(" Boxplot that visualizes the distribution of the damt target variable and the wrat variable ")


plt.figure(figsize=(8, 6))
sns.boxplot(x='wrat', y='damt', data=filtered_data)

plt.xlabel('wrat (0 - 9)')
plt.ylabel('damt')
plt.title('Boxplot of damt by wrat')
st.pyplot(plt)


# In[ ]:


st.header("Violin Plot: damt & wrat")
st.write("Violinplot that visualizes the density and underlying distribution of the damt target variable and the wrat variable ")

plt.figure(figsize=(25, 20)) 
sns.violinplot(x='wrat', y='damt', data=filtered_data)
plt.xlabel('wrat (0 - 9)')
plt.ylabel('damt')
plt.title('Violin Plot of damt by wrat')
st.pyplot(plt)


# In[ ]:


st.header("Key Insights")
st.write("""
    - **Insight 1:** Based on the correlation matrix and heatmap created, the independent variables of “home” and "wrat" had the highest positive correlation to the target variable of “damt”; this indicates that increases in the “home” and the "wrat" variables are associated with increases in the “damt” variable.
    - **Insight 2:** The Boxplot indicates that the data is left skewed due to the median line being closer to the higher end of the box; this indicates that the data is more concentrated towards high values of "damt", but there is a long whisker that can be seen which suggests a large spread. 
    - **Insight 3:** The Violinplot’s width demonstrates that for those with a home, the values at 13,000 damt are frequent while for those without a home,the values are concentrated frequently at 0 damt.
    - **Insight 4:** For values of "wrat" from categories 5-9, there are larger variations in donation amounts in the Violinplot, suggesting that donors with higher wealth ratings give more donations but with greater variability.
    - **Insight 5:** As "wrat" increases from 5, the median donation increases in the Boxplot; this also displays that the highest donations occur from categories 6-9. 
""")
st.markdown("""
    **Key Findings:**
    - Those who own homes, donate larger amounts than those who do not own homes. 
    - Those with high wealth ratings (ranging from 5-9), donate larger amounts than those with low wealth ratings.
""")


# In[ ]:


st.header("Recommendations")
st.write("""
    Based on the analysis of the charitydata dataset, we would recommend the following actions:
    1. Tailor marketing efforts towards those with a wealth rating between categories 5 and 9.
    2. Focus marketing efforts around community benefits revolving around the donations from those who own homes.
    3. Highlight long term outcomes in marketing efforts that would impact those who own homes and have a wealth rating between 5 and 9. 
""")
st.markdown("""
    **Next Steps:**
    - Run targeted campaigns for homeowners with a wealth rating between 5 and 9 to encourage further donations.
    - Utilize email campaigns to applaud and recognize those who have donated in the past in order to engage with the audience and further strengthen relationships.
""")

