#!/usr/bin/env python
# coding: utf-8

'''
    All the code written in this project has been created and tested in a Jupyter Notebook environment.
'''

# **Automatidata project**
#     **Course 2**

# Welcome to the Automatidata Project!
# You have just started as a data professional in a fictional data consulting firm, Automatidata. 
# Their client, the New York City Taxi and Limousine Commission (New York City TLC), has hired the Automatidata team for its reputation in helping their clients develop data-based solutions.

# The team is still in the early stages of the project. Previously, you were asked to complete a project proposal by your supervisor, DeShawn Washington. 
# You have received notice that your project proposal has been approved and that New York City TLC has given the Automatidata team access to their data. 
# To get clear insights, New York TLC's data must be analyzed, key variables identified, and the dataset ensured it is ready for analysis.

# A notebook was structured and prepared to help you in this project. Please complete the following questions.

  # Course 2 End-of-course project: Inspect and analyze data

# In this activity, you will examine data provided and prepare it for analysis.  This activity will help ensure the information is,

# 1.   Ready to answer questions and yield insights
# 2.   Ready for visualizations
# 3.   Ready for future hypothesis testing and statistical methods 

# **The purpose** of this project is to investigate and understand the data provided.
# **The goal** is to use a dataframe contructed within Python, perform a cursory inspection of the provided dataset, and inform team members of your findings.

# *This activity has three parts:*
# **Part 1:** Understand the situation 
# * Prepare to understand and organize the provided taxi cab dataset and information.

# Part 2: Import necessary libraries
import pandas as pd
import numpy as np


# * Create a pandas dataframe for data learning, future exploratory data analysis (EDA), and statistical activities.
# * Compile summary information about the data to inform next steps.
# **Part 3:** Understand the variables
# * Use insights from your examination of the summary data to guide deeper investigation into specific variables.

# Follow the instructions and answer the following questions to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work. 

# # **Identify data types and relevant variables using Python**

# # **PACE stages**

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.
# ## PACE: **Plan**
# 
# Consider the questions in your PACE Strategy Document and those below to craft your response:

# **Task 1. Understand the situation**

# ## PACE: **Analyze**

# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### **Task 2a. Build dataframe**

# Create a pandas dataframe for data learning, and future exploratory data analysis (EDA) and statistical activities.

# Create and load pandas dataframe
df = pd.read_csv('2017_Yellow_Taxi_Trip_Data.csv')
print("done")

# ### **Task 2b. Understand the data - Inspect the data**
# View and inspect summary information about the dataframe by coding the following:
# 
# 1. `df.head(10)`
# 2. `df.info()`
# 3. `df.describe()`
# 
# Consider the following two questions:
# 
# **Question 1:** When reviewing the `df.info()` output, what do you notice about the different variables? Are there any null values? Are all of the variables numeric? Does anything else stand out?
# 
# **Question 2:** When reviewing the `df.describe()` output, what do you notice about the distributions of each variable? Are there any questionable values?

# 1. No null values, almost all numeric, int, float and objects. store_and_fwd_flag could be boolean instead of object for better processing.

# 2. Found negative and extreme values in the minimum and maximum amount:
# RatecodeID: has a maximum value of 99, when it should be maximum 6.
# payment_type: has a maximum value of 4, when it should be maximum 2.
# fare_amount: has a minimum of -120.

df.head(10)
df.info()
df.describe()

# ### **Task 2c. Understand the data - Investigate the variables**
# 
# Sort and interpret the data table for two variables:`trip_distance` and `total_amount`.
# 
# **Answer the following three questions:**
# 
# **Question 1:** Sort your first variable (`trip_distance`) from maximum to minimum value, do the values seem normal?
# 
# **Question 2:** Sort by your second variable (`total_amount`), are any values unusual?
# 
# **Question 3:** Are the resulting rows similar for both sorts? Why or why not?

# 1. Some values are 0.
# 
# 2. Yes, there are a lot of unusual values.
# Negative values: they don't make sense.
# Extreme values: they don't match with the distance traveled (fare it's too high).
# 
# 3. There are trips with 0 distance and wrong fares, creating unusual values.

sorted_df = df.sort_values(by='trip_distance', ascending=False)
sorted_df = df.sort_values(by='total_amount', ascending=False)
print(sorted_df.head(20))

# Sort the data by total amount and print the top 20 values
sorted_df = df.sort_values(by='total_amount', ascending=False)
print(sorted_df.tail(20))

payment_counts = df['payment_type'].value_counts()
print(payment_counts)

# According to the data dictionary, the payment method was encoded as follows:
# 
# 1 = Credit card  
# 2 = Cash  
# 3 = No charge  
# 4 = Dispute  
# 5 = Unknown  
# 6 = Voided trip

tips_credit_card = df[df['payment_type'] == 1]
avg_tip_credit_card = tips_credit_card['tip_amount'].mean()
print(avg_tip_credit_card)

tips_cash = df[df['payment_type'] == 2]
avg_tip_cash = tips_cash['tip_amount'].mean()
print(avg_tip_cash)

vendorID_counts = df['VendorID'].value_counts()
print(vendorID_counts)

# How many times is each vendor ID represented in the data?

mean_total_per_vendor = df.groupby('VendorID')['total_amount'].mean()
print(mean_total_per_vendor)

# What is the mean total amount for each vendor?

credit_cards = df[df['payment_type'] == 1]
print(credit_cards)
# Filter the data for credit card payments only

passenger_count_only = credit_cards[['passenger_count']]
print(passenger_count_only)
# Filter the credit-card-only data for passenger count only

tips_credit_card = df[df['payment_type'] == 1]
avg_tip_credit_card = tips_credit_card['tip_amount'].mean()

passenger_count_only = credit_cards[['passenger_count']]

average_tip_per_passenger = credit_cards.groupby('passenger_count')['tip_amount'].mean().reset_index()
print(average_tip_per_passenger)
# Calculate the average tip amount for each passenger count (credit card payments only)

# ## PACE: **Construct**
# 
# **Note**: The Construct stage does not apply to this workflow. The PACE framework can be adapted to fit the specific requirements of any project. 

# ## PACE: **Execute**
# Consider the questions in your PACE Strategy Document and those below to craft your response.

# ### **Given your efforts, what can you summarize for DeShawn and the data team?**
# *Note for Learners: Your notebook should contain data that can address Luana's requests. Which two variables are most helpful for building a predictive model for the client: NYC TLC?*

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. 
# Just click on the "save" icon at the top of this notebook to ensure your work has been logged.