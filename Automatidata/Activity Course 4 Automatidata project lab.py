#!/usr/bin/env python
# coding: utf-8

'''
    All the code written in this project has been created and tested in a Jupyter Notebook environment.
'''

# # Automatidata project 
# **Course 4 - The Power of Statistics**

# You are a data professional in a data consulting firm, called Automatidata. The current project for their newest client, the New York City Taxi & Limousine Commission (New York City TLC) is reaching its midpoint, having completed a project proposal, Python coding work, and exploratory data analysis.
# 
# You receive a new email from Uli King, Automatidata’s project manager. Uli tells your team about a new request from the New York City TLC: to analyze the relationship between fare amount and payment type. 
# A follow-up email from Luana includes your specific assignment: to conduct an A/B test. 
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # Course 4 End-of-course project: Statistical analysis
# 
# In this activity, you will practice using statistics to analyze and interpret data. The activity covers fundamental concepts such as descriptive statistics and hypothesis testing. 
# You will explore the data provided and conduct A/B and hypothesis testing.  
# **The purpose** of this project is to demostrate knowledge of how to prepare, create, and analyze A/B tests. Your A/B test results should aim to find ways to generate more revenue for taxi cab drivers.
# **Note:** For the purpose of this exercise, assume that the sample data comes from an experiment in which customers are randomly selected and divided into two groups: 
# 1) customers who are required to pay with credit card.
# 2) customers who are required to pay with cash. Without this assumption, we cannot draw causal conclusions about how payment method affects fare amount.
# **The goal** is to apply descriptive statistics and hypothesis testing in Python. The goal for this A/B test is to sample data and analyze whether there is a relationship between payment type and fare amount. For example: discover if customers who use credit cards pay higher fare amounts than customers who use cash.
#
# *This activity has four parts:*
#
# **Part 1:** Imports and data loading
# * What data packages will be necessary for hypothesis testing?
#
# **Part 2:** Conduct EDA and hypothesis testing
# * How did computing descriptive statistics help you analyze your data? 
#
# * How did you formulate your null hypothesis and alternative hypothesis? 
#
# **Part 3:** Communicate insights with stakeholders
#
# * What key business insight(s) emerged from your A/B test?
#
# * What business recommendations do you propose based on your results?
#
# Follow the instructions and answer the questions below to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.
#
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work. 

# # **Conduct an A/B test**
# # **PACE stages**

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# ## PACE: Plan 
# 
# In this stage, consider the following questions where applicable to complete your code response:
# 1. What is your research question for this data project? Later on, you will need to formulate the null and alternative hypotheses as the first step of your hypothesis test. 
# Consider your research question now, at the start of this task.

# Is there a relationship between total fare amount and payment type?

# *Complete the following steps to perform statistical analysis of your data:* 

# ### Task 1. Imports and data loading

# Import packages and libraries needed to compute descriptive statistics and conduct a hypothesis test.
# In[1]:
import pandas as pd
from scipy import stats

# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. 
# Please continue with this activity by completing the following instructions.

# In[2]:
# Load dataset into dataframe
taxi_data = pd.read_csv("2017_Yellow_Taxi_Trip_Data.csv", index_col = 0)

# ## PACE: **Analyze and Construct**
# In this stage, consider the following questions where applicable to complete your code response:
# 1. Data professionals use descriptive statistics for Exploratory Data Analysis. How can computing descriptive statistics help you learn more about your data in this stage of your analysis?

# Descriptive statistics are useful because they let you quickly explore and understand large amounts of data. 
# In this case, computing descriptive statistics helps you quickly compare the average total fare amount among different payment types.

# ### Task 2. Data exploration
# 
# Use descriptive statistics to conduct Exploratory Data Analysis (EDA). 

# **Note:** In the dataset, `payment_type` is encoded in integers:
# *   1: Credit card
# *   2: Cash
# *   3: No charge
# *   4: Dispute
# *   5: Unknown

# In[4]:
#Describe
taxi_data.describe(include='all')

# You are interested in the relationship between payment type and the fare amount the customer pays. One approach is to look at the average fare amount for each payment type. 

# In[5]:
taxi_data.groupby('payment_type')['fare_amount'].mean()

# Based on the averages shown, it appears that customers who pay in credit card tend to pay a larger fare amount than customers who pay in cash. 
# However, this difference might arise from random sampling, rather than being a true difference in fare amount. To assess whether the difference is statistically significant, you conduct a hypothesis test.

# ### Task 3. Hypothesis testing
# 
# Before you conduct your hypothesis test, consider the following questions where applicable to complete your code response:
# 
# 1. Recall the difference between the null hypothesis and the alternative hypotheses. Consider your hypotheses for this project as listed below.
# 
# **Null hypothesis**: There is no difference in average fare between customers who use credit cards and customers who use cash. 
# 
# **Alternative hypothesis**: There is a difference in average fare between customers who use credit cards and customers who use cash
# 
# $H_0$: There is no difference in the average fare amount between customers who use credit cards and customers who use cash.
# 
# $H_A$: There is a difference in the average fare amount between customers who use credit cards and customers who use cash.

# Your goal in this step is to conduct a two-sample t-test. Recall the steps for conducting a hypothesis test: 
# 
# 
# 1.   State the null hypothesis and the alternative hypothesis
# 2.   Choose a signficance level
# 3.   Find the p-value
# 4.   Reject or fail to reject the null hypothesis 

# **Note:** For the purpose of this exercise, your hypothesis test is the main component of your A/B test.

# You choose 5% as the significance level and proceed with a two-sample t-test.

# In[6]:
#hypothesis test, A/B test
#significance level
credit_card = taxi_data[taxi_data['payment_type'] == 1]['fare_amount']
cash = taxi_data[taxi_data['payment_type'] == 2]['fare_amount']
stats.ttest_ind(a=credit_card, b=cash, equal_var=False)

# We reject the null hypothesis because the p value it's smaller than the significance level.

# ## PACE: **Execute**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### Task 4. Communicate insights with stakeholders

# *Ask yourself the following questions:*
# 
# 1. What business insight(s) can you draw from the result of your hypothesis test?
# 2. Consider why this A/B test project might not be realistic, and what assumptions had to be made for this educational project.

# 1.   Encouraging customers to pay with credit cards can generate more revenue for taxi cab drivers.
# 
# 2.   This project requires an assumption that passengers were forced to pay one way or the other, and that once informed of this requirement, they always complied with it. 
#      The data was not collected this way; so, an assumption had to be made to randomly group data entries to perform an A/B test. 
#      This dataset does not account for other likely explanations. For example, riders might not carry lots of cash, so it's easier to pay for longer/farther trips with a credit card. 
#      In other words, it's far more likely that fare amount determines payment type, rather than vice versa. 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.