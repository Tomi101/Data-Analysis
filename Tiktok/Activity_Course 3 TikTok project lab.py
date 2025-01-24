#!/usr/bin/env python
# coding: utf-8

'''
    All the code written in this project has been created and tested in a Jupyter Notebook environment.
'''

# # **TikTok Project**
# **Course 3 - Go Beyond the Numbers: Translate Data into Insights**

# Your TikTok data team is still in the early stages of their latest project. So far, you’ve completed a project proposal and used Python to inspect and organize the TikTok dataset.
# 
# Orion Rainier, a Data Scientist at TikTok, is pleased with the work you have already completed and is requesting your assistance with some Exploratory Data Analysis (EDA) and data visualization. The management team asked to see a Python notebook showing data structuring and cleaning, as well as any matplotlib/seaborn visualizations plotted to help us understand the data. At the very least, include a graph comparing claim counts to opinion counts, as well as boxplots of the most important variables (like “video duration,” “video like count,” “video comment count,” and “video view count”) to check for outliers. Also, include a breakdown of “author ban status” counts.
# 
# Additionally, the management team has recently asked all EDA to include Tableau visualizations. Tableau visualizations are particularly helpful in status reports to the client and board members. For this data, create a Tableau dashboard showing a simple claims versus opinions count, as well as stacked bar charts of claims versus opinions for variables like video view counts, video like counts, video share counts, and video download counts. Make sure it is easy to understand to someone who isn’t data savvy, and remember that the assistant director is a person with visual impairments.
# 
# You also notice a follow-up email from the Data Science Lead, Willow Jaffey. Willow suggests including an executive summary of your analysis to share with teammates.
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # **Course 3 End-of-course project: Exploratory data analysis**
# 
# In this activity, you will examine data provided and prepare it for analysis. You will also design a professional data visualization that tells a story, and will help data-driven decisions for business needs.
# 
# Please note that the Tableau visualization activity is optional, and will not affect your completion of the course. Completing the Tableau activity will help you practice planning out and plotting a data visualization based on a specific business need. The structure of this activity is designed to emulate the proposals you will likely be assigned in your career as a data professional. Completing this activity will help prepare you for those career moments.
# **The purpose** of this project is to conduct exploratory data analysis on a provided data set. Your mission is to continue the investigation you began in C2 and perform further EDA on this data with the aim of learning more about the variables. Of particular interest is information related to what distinguishes claim videos from opinion videos.
# 
# **The goal** is to explore the dataset and create visualizations.
# *This activity has 4 parts:*
# 
# **Part 1:** Imports, links, and loading
# 
# **Part 2:** Data Exploration
# *   Data cleaning
# 
# **Part 3:** Build visualizations
# 
# **Part 4:** Evaluate and share results

# Follow the instructions and answer the question below to complete the activity. Then, you will complete an executive summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.

# # **Visualize a story in Tableau and Python**

# # **PACE stages**

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# ## **PACE: Plan**
# 
# Consider the questions in your PACE Strategy Document and those below where applicable to craft your response:
# 1. Identify any outliers:
# 
# *   What methods are best for identifying outliers?
# *   How do you make the decision to keep or exclude outliers from any future models?

# Using Numpy Functions:
#   * Utilize functions like mean() and median() to understand the central tendency of the data and assess the range of data values. This can help identify values that deviate significantly from the norm.
# 
# Boxplots for Visualization:
#  * Create a boxplot to visualize the data's distribution and easily spot potential outliers as points outside the whiskers.

# ### **Task 1. Imports, links, and loading**
# Go to Tableau Public
# The following link will help you complete this activity. Keep Tableau Public open as you proceed to the next steps.
# 
# Link to supporting materials:
# Public Tableau: https://public.tableau.com/s/. Note that the TikTok dataset can be downloaded directly from this notebook by going to "Lab Files" in the menu bar at the top of the page, clicking into the "/home/jovyan/work" folder, selecting `tiktok_dataset.csv`, and clicking "Download" above the list of files. 
# 
# For EDA of the data, import the packages that would be most helpful, such as `pandas`, `numpy`, `matplotlib.pyplot`, and `seaborn`.

# In[76]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Then, load the dataset into a dataframe. Read in the data and store it as a dataframe object.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[77]:
# Load dataset into dataframe
df = pd.read_csv("tiktok_dataset.csv")

# ## **PACE: Analyze**
# 
# Consider the questions in your PACE Strategy Document and those below where applicable to complete your code.

# ### **Task 2a: Data exploration and cleaning**
# 
# The first step is to assess your data. Check the Data Source page on Tableau Public to get a sense of the size, shape and makeup of the data set.
# 
# Consider functions that help you understand and structure the data.
# 
# *    `.head()`
# *    `.info()`
# *    `.describe()`
# *    `.groupby()`
# *    `.sort_values()`
# 
# Consider the following questions as you work:
# 
# What do you do about missing data (if any)?
# 
# Are there data outliers?

# Start by discovering, using `.head()`, `.size`, and `.shape`.

# In[9]:
# Display and examine the first few rows of the dataframe
df.head()

# In[10]:
# Get the size of the data
df.size

# In[6]:
# Get the shape of the data
data.shape

# In[11]:
# Get basic information about the data.
df.info()

# In[12]:
# Generate a table of descriptive statistics
df.describe()

# ### **Task 2b. Assess data types**

# In Tableau, staying on the data source page, double check the data types of the columns in the dataset. Refer to the dimensions and measures in Tableau.

# Review the instructions linked in the previous Activity document to create the required Tableau visualization.

# ### **Task 2c. Select visualization type(s)**

# Select data visualization types that will help you understand and explain the data.
# 
# Now that you know which data columns you’ll use, it is time to decide which data visualization makes the most sense for EDA of the TikTok dataset. What type of data visualization(s) would be most helpful? Consider the distribution of the data.
# 
# * Line graph
# * Bar chart
# * Box plot
# * Histogram
# * Heat map
# * Scatter plot
# * A geographic map

# Box plots and histograms. Visualizing the distribution of the data can inform the next steps and considerations.

# ## **PACE: Construct**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.

# ### **Task 3. Build visualizations**
# 
# Now that you have assessed your data, it’s time to plot your visualization(s).

# #### **video_duration_sec**
# 
# Create a box plot to examine the spread of values in the `video_duration_sec` column.

# In[78]:
# Create a boxplot to visualize distribution of `video_duration_sec`
plt.figure(figsize=(5,1))
plt.title('video_duration_sec')
sns.boxplot(x=df['video_duration_sec']);

# Create a histogram of the values in the `video_duration_sec` column to further explore the distribution of this variable.

# In[79]:
plt.figure(figsize=(5,3))
sns.histplot(df['video_duration_sec'], bins=range(0,61,5))
plt.title('Video duration histogram');

# **Question:** What do you notice about the duration and distribution of the videos?
# 
# All videos are 5-60 seconds in length, and the distribution is uniform.

# #### **video_view_count**
# 
# Create a box plot to examine the spread of values in the `video_view_count` column.

# In[80]:
# Create a boxplot to visualize distribution of `video_view_count`
plt.figure(figsize=(5, 1))
plt.title('video_view_count')
sns.boxplot(x=df['video_view_count']);

# Create a histogram of the values in the `video_view_count` column to further explore the distribution of this variable.

# In[81]:
plt.figure(figsize=(5,3))
sns.histplot(df['video_view_count'], bins=range(0,(10**6+1),10**5))
plt.title('Video view count histogram');

# **Question:** What do you notice about the distribution of this variable?
# 
# This variable has a very uneven distribution, with more than half the videos receiving fewer than 100,000 views. Distribution of view counts > 100,000 views is uniform.

# #### **video_like_count**
# 
# Create a box plot to examine the spread of values in the `video_like_count` column.

# In[82]:
# Create a boxplot to visualize distribution of `video_like_count`
plt.figure(figsize=(10,1))
plt.title('video_like_count')
sns.boxplot(x=df['video_like_count']);

# Create a histogram of the values in the `video_like_count` column to further explore the distribution of this variable.

# In[83]:
# plt.figure(figsize=(5,3))
ax = sns.histplot(df['video_like_count'], bins=range(0,(7*10**5+1),10**5))
labels = [0] + [str(i) + 'k' for i in range(100, 701, 100)]
ax.set_xticks(range(0,7*10**5+1,10**5), labels)
plt.title('Video like count histogram');

# **Question:** What do you notice about the distribution of this variable?
# 
# Similar to view count, there are far more videos with < 100,000 likes than there are videos with more. However, in this case, there is more of a taper, as the data skews right, with many videos at the upper extremity of like count.

# In[84]:
# Create a box plot to examine the spread of values in the `video_comment_count` column.
plt.figure(figsize=(5,1))
plt.title('video_comment_count')
sns.boxplot(x=df['video_comment_count']);

# In[85]:
# Create a histogram of the values in the `video_comment_count` column to further explore the distribution of this variable.
plt.figure(figsize=(10,5))
sns.histplot(df['video_comment_count'], bins=range(0,(3001),100))
plt.title('Video comment count histogram');

# **Question:** What do you notice about the distribution of this variable?
# Again, the vast majority of videos are grouped at the bottom of the range of values for video comment count. Most videos have fewer than 100 comments. The distribution is very right-skewed.

# In[86]:
# Create a box plot to examine the spread of values in the `video_share_count` column.
plt.figure(figsize=(5,1))
plt.title('video_share_count')
sns.boxplot(x=df['video_share_count']);

# In[87]:
# *Create* a histogram of the values in the `video_share_count` column to further explore the distribution of this variable.
plt.figure(figsize=(5,3))
sns.histplot(df['video_share_count'], bins=range(0,(270001),10000))
plt.title('Video share count histogram');

# **Question:** What do you notice about the distribution of this variable?
# 
# The majority of videos had fewer than 10,000 shares. The distribution is very skewed to the right.

# In[88]:
# Create a box plot to examine the spread of values in the `video_download_count` column.
plt.figure(figsize=(5,1))
plt.title('video_download_count')
sns.boxplot(x=df['video_download_count']);

# In[89]:
# Create a histogram of the values in the `video_download_count` column to further explore the distribution of this variable.
plt.figure(figsize=(5,3))
sns.histplot(df['video_download_count'], bins=range(0,(15001),500))
plt.title('Video download count histogram');


# **Question:** What do you notice about the distribution of this variable?
# 
# The majority of videos were downloaded fewer than 500 times, but some were downloaded over 12,000 times. Again, the data is very skewed to the right.

# In[90]:
# Create a histogram with four bars: one for each combination of claim status and verification status.
plt.figure(figsize=(7,4))
sns.histplot(data=df,
             x='claim_status',
             hue='verified_status',
             multiple='dodge',
             shrink=0.9)
plt.title('Claims by verification status histogram');

# **Question:** What do you notice about the number of verified users compared to unverified? And how does that affect their likelihood to post opinions?
# 
# There are fewer verified users than unverified users, but if a user *is* verified, they are much more likely to post opinions.

# #### **Claim status by author ban status**
# 
# The previous course used a `groupby()` statement to examine the count of each claim status for each author ban status. Now, use a histogram to communicate the same information.

# In[91]:
# Create a histogram
fig = plt.figure(figsize=(7,4))
sns.histplot(data, x='claim_status', hue='author_ban_status',
             multiple='dodge',
             hue_order=['active', 'under review', 'banned'],
             shrink=0.9,
             palette={'active':'green', 'under review':'orange', 'banned':'red'},
             alpha=0.5)
plt.title('Claim status by author ban status - counts');

# **Question:** What do you notice about the number of active authors compared to banned authors for both claims and opinions?
# 
# For both claims and opinions, there are many more active authors than banned authors or authors under review; however, the proportion of active authors is far greater for opinion videos than for claim videos. Again, it seems that authors who post claim videos are more likely to come under review and/or get banned.

# #### **Median view counts by ban status**
# 
# Create a bar plot with three bars: one for each author ban status. The height of each bar should correspond with the median number of views for all videos with that author ban status.

# In[92]:
# Create a bar plot
ban_status_counts = df.groupby(['author_ban_status']).median(
    numeric_only=True).reset_index()

fig = plt.figure(figsize=(5,3))
sns.barplot(data=ban_status_counts,
            x='author_ban_status',
            y='video_view_count',
            order=['active', 'under review', 'banned'],
            palette={'active':'green', 'under review':'orange', 'banned':'red'},
            alpha=0.5)
plt.title('Median view count by ban status');

# **Question:** What do you notice about the median view counts for non-active authors compared to that of active authors? Based on that insight, what variable might be a good indicator of claim status?
# 
# The median view counts for non-active authors are many times greater than the median view count for active authors. Since you know that non-active authors are more likely to post claims, and that videos by non-active authors get far more views on aggregate than videos by active authors, then `video_view_count` might be a good indicator of claim status.
# 
# Indeed, a quick check of the median view count by claim status bears out this assessment:

# In[93]:
# Calculate the median view count for claim status.
df.groupby('claim_status')['video_view_count'].median()


# #### **Total views by claim status**
# 
# Create a pie graph that depicts the proportions of total views for claim videos and total views for opinion videos.

# In[94]:
fig = plt.figure(figsize=(5,5))
plt.pie(df.groupby('claim_status')['video_view_count'].sum(), labels=['claim', 'opinion'])
plt.title('Total views by video claim status');

# **Question:** What do you notice about the overall view count for claim status?
# 
# The overall view count is dominated by claim videos even though there are roughly the same number of each video in the dataset.

# ### **Task 4. Determine outliers**
# 
# When building predictive models, the presence of outliers can be problematic. For example, if you were trying to predict the view count of a particular video, videos with extremely high view counts might introduce bias to a model. Also, some outliers might indicate problems with how data was captured or recorded.
# 
# The ultimate objective of the TikTok project is to build a model that predicts whether a video is a claim or opinion. The analysis you've performed indicates that a video's engagement level is strongly correlated with its claim status. There's no reason to believe that any of the values in the TikTok data are erroneously captured, and they align with expectation of how social media works: a very small proportion of videos get super high engagement levels. That's the nature of viral content.
# 
# Nonetheless, it's good practice to get a sense of just how many of your data points could be considered outliers. The definition of an outlier can change based on the details of your project, and it helps to have domain expertise to decide a threshold. You've learned that a common way to determine outliers in a normal distribution is to calculate the interquartile range (IQR) and set a threshold that is 1.5 * IQR above the 3rd quartile.
# 
# In this TikTok dataset, the values for the count variables are not normally distributed. They are heavily skewed to the right. One way of modifying the outlier threshold is by calculating the **median** value for each variable and then adding 1.5 * IQR. This results in a threshold that is, in this case, much lower than it would be if you used the 3rd quartile.
# 
# Write a for loop that iterates over the column names of each count variable. For each iteration:
# 1. Calculate the IQR of the column
# 2. Calculate the median of the column
# 3. Calculate the outlier threshold (median + 1.5 * IQR)
# 4. Calculate the numer of videos with a count in that column that exceeds the outlier threshold
# 5. Print "Number of outliers, {column name}: {outlier count}"
# 
# ```
# Example:
# Number of outliers, video_view_count: ___
# Number of outliers, video_like_count: ___
# Number of outliers, video_share_count: ___
# Number of outliers, video_download_count: ___
# Number of outliers, video_comment_count: ___
# ```

# In[95]:
count_cols = ['video_view_count',
              'video_like_count',
              'video_share_count',
              'video_download_count',
              'video_comment_count',
              ]

for column in count_cols:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    median = df[column].median()
    outlier_threshold = median + 1.5*iqr

    # Count the number of values that exceed the outlier threshold
    outlier_count = (df[column] > outlier_threshold).sum()
    print(f'Number of outliers, {column}:', outlier_count)


# #### **Scatterplot**

# In[96]:
# Create a scatterplot of `video_view_count` versus `video_like_count` according to 'claim_status'
sns.scatterplot(x=df["video_view_count"], y=df["video_like_count"],
                hue=df["claim_status"], s=10, alpha=.3)
plt.show()

# In[97]:
# Create a scatterplot of `video_view_count` versus `video_like_count` for opinions only
opinion = df[df['claim_status']=='opinion']
sns.scatterplot(x=opinion["video_view_count"], y=opinion["video_like_count"],
                 s=10, alpha=.3)
plt.show()

# You can do a scatterplot in Tableau Public as well, which can be easier to manipulate and present. If you'd like step by step instructions, you can review the instructions linked in the previous Activity page.
# ## **PACE: Execute**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### **Task 5a. Results and evaluation**
# 
# Having built visualizations in Tableau and in Python, what have you learned about the dataset? What other questions have your visualizations uncovered that you should pursue?
# 
# ***Pro tip:*** Put yourself in your client's perspective, what would they want to know?
# 
# Use the following code cells to pursue any additional EDA. Also use the space to make sure your visualizations are clean, easily understandable, and accessible.
# 
# ***Ask yourself:*** Did you consider color, contrast, emphasis, and labeling?

# I have learned ....
# 
#  * How to examine the data distribution/spread, count frequencies, mean and median values, extreme values/outliers, missing data, and more. I analyzed correlations between variables, particularly between the claim_status variable and others.
# 
# My other questions are ....
# 
#  * I want to further investigate distinctive characteristics that apply only to claims or only to opinions. Also, I want to consider other variables that might be helpful in understanding the data.
# 
# My client would likely want to know ...
# 
#  * My client would want to know the assumptions regarding what data might be predictive of claim_status.

# ### **Task 5b. Conclusion**
# *Make it professional and presentable*
# 
# You have visualized the data you need to share with the director now. Remember, the goal of a data visualization is for an audience member to glean the information on the chart in mere seconds.
# 
# *Questions to ask yourself for reflection:*
# Why is it important to conduct Exploratory Data Analysis? What other visuals could you create?

# EDA is important because ...
# 
# * EDA it's important to get to know the data, understand its outliers, clean its missing values, and prepare it for future modeling.
# 
# Visualizations helped me understand ..
# 
# * That we will need to make decisions on certain considerations prior to designing a model. For example, what to do with outliers, duplicate values, or missing data.

# You’ve now completed a professional data visualization according to a business need. Well done! Be sure to save your work as a reference for later work in Tableau.

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.