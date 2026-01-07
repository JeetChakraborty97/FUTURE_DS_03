# FUTURE_DS_03
Repository for Task 3 of the Data Science &amp; Analytics Internship at Future Interns

# Student Feedback Analysis

<img width="1536" height="1024" alt="ChatGPT Image Jan 7, 2026, 08_42_10 PM" src="https://github.com/user-attachments/assets/cf7c480a-197f-42e0-8d15-f760d199b357" />

# Overview
## Project Overview

The objective of this project is to analyse student feedback data collected after academic courses and learning experiences, identify satisfaction patterns, and extract actionable insights that can help improve teaching quality and overall student experience.

The analysis focuses on: 
* Understanding rating trends across multiple feedback parameters
* Measuring overall satisfaction levels
* Deriving sentiment categories based on quantitative feedback
* Providing data-driven recommendations for improvement

## Dataset Overview

* Dataset: Student Feedback Ratings (CSV)

Link: https://www.kaggle.com/datasets/ruchi798/student-feedback-survey-responses
* Records Analysed: 1001 student responses

The dataset consists of numerical ratings provided by students on a scale of 1 to 10 for the following aspects: 
* Subject Knowledge
* Teaching Clarity
* Use of Presentations
* Assignment Difficulty
* Doubt Solving
* Course Structure
* Student Support
* Course Recommendation 

Each row represents feedback from one student.

## Tools Used

* Python: Pandas, Numpy, Matplotlib, Seaborn.
* Jupyter Notebook: Workspace.

## Step-by-Step Process (With Code and Visuals)

### Importing all the necessary libraries & Other Necessary Things.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set()
```
### Loading the dataset.
```python
df = pd.read_csv("D:\Career\Internships\Future Interns\Task 3\Dataset\student_feedback.csv")
df.head()
```
Here is the snippet of the dataset loaded:
<img width="1169" height="245" alt="1" src="https://github.com/user-attachments/assets/e03bbb14-8689-43f4-b620-6e250059045b" />

### Understanding the dataset structure.

First Step:

```python
df.info()
```
Here is the output:
<img width="1187" height="290" alt="2" src="https://github.com/user-attachments/assets/a8a54065-7054-4edc-bffa-3b102a75d4b6" />

Second Step:

```python
df.describe()
```
Here is the output:
<img width="1182" height="358" alt="3" src="https://github.com/user-attachments/assets/07a21d8b-7ab2-4f21-b129-934d42b7672a" />

### Moving to Data Cleaning.

Dropping Unnecessary Column: The column "Unnamed:" 0 is just an index. So, I've decided to drop it.

```python
df.drop(columns=['Unnamed: 0'], inplace=True)
```

Checking missing values.

```python
df.isnull().sum()
```
Here is the output:
<img width="1169" height="177" alt="4" src="https://github.com/user-attachments/assets/5f7dee7a-f177-4649-8f66-dd30bffb38d2" />

Result: No missing values found.

### Moving to Overall Satisfaction Analysis.

* Creating an Overall Satisfaction Score.

```python
df['Overall_Satisfaction'] = df.iloc[:, 1:].mean(axis=1)
df.head()
```
Here is the output:
<img width="1179" height="261" alt="5" src="https://github.com/user-attachments/assets/7e277faa-5719-48dc-b5b3-4cd2eecdf0eb" />

* Finding the distribution of Overall Satisfaction.

```python
plt.figure(figsize=(8,5))
plt.hist(df['Overall_Satisfaction'], bins=10)
plt.xlabel("Overall Satisfaction Score")
plt.ylabel("Number of Students")
plt.title("Distribution of Overall Satisfaction")
plt.show()
```
Here is the output:
<img width="1165" height="483" alt="6" src="https://github.com/user-attachments/assets/403b64d9-53ed-43c6-b1d4-3cbda329d5e0" />

Insight: Most of the satisfaction scores are clustered around the middle point. So, most of the students have a moderate satisfaction rate.

### Moving on to Average Rating per Parameter.
```python
avg_ratings = df.iloc[:, 1:-1].mean().sort_values(ascending=False)
avg_ratings
```
Here is the output:
<img width="1159" height="167" alt="7" src="https://github.com/user-attachments/assets/52ecc983-cab2-48fc-90a2-089aa1e99169" />

* Providing the visualisation for it with a Bar Chart.

```python
plt.figure(figsize=(10,6))
avg_ratings.plot(kind='bar')
plt.ylabel("Average Rating")
plt.title("Average Ratings by Feedback Category")
plt.xticks(rotation=45, ha='right')
plt.show()
```
Here is the output:
<img width="1162" height="785" alt="8" src="https://github.com/user-attachments/assets/f4c41a69-d32b-4528-b5b2-9cb352d38dc5" />

Insight: From the chart, we can see that students have rated the experience of the teacher as the highest among all the categories. Other categories have received fairly similar ratings.

### Moving on to Satisfaction Level Classification.

* Creating Satisfaction Labels.

```python
def satisfaction_label(score):
    if score >= 7:
        return "Positive"
    elif score >= 4:
        return "Neutral"
    else:
        return "Negative"

df['Satisfaction_Level'] = df['Overall_Satisfaction'].apply(satisfaction_label)
```

* Creating Distribution of Satisfaction Levels.

```python
df['Satisfaction_Level'].value_counts()
```
Here is the output:
<img width="1181" height="95" alt="9" src="https://github.com/user-attachments/assets/3cff7b17-bacd-4d52-9317-38d69309083a" />

* Creating Pie Chart visualization.

```python
df['Satisfaction_Level'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    figsize=(6,6),
    title="Student Satisfaction Levels"
)
plt.ylabel("")
plt.show()
```
Here is the output:
<img width="1172" height="514" alt="10" src="https://github.com/user-attachments/assets/ac14b80b-f254-43c0-b759-4ed2ee07b5ea" />

Insight: Most of the students are showing 'Neutral' satisfaction level and signaling a neutral sentiment.

### Moving on to Correlation Analysis.
```python
plt.figure(figsize=(10,6))
sns.heatmap(df.iloc[:,1:-2].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Feedback Parameters")
plt.show()
```
Here is the output:
<img width="1057" height="790" alt="11" src="https://github.com/user-attachments/assets/e8ac0417-920b-44ce-89f8-89ed6de577d3" />

Insight: There is a strong link between clarity of explanation and course structure.

### Moving on to Top & Weak Areas Identification.

* Finding top 3 strengths.

```python
avg_ratings.head(3)
```
Here is the output:
<img width="1052" height="72" alt="12" src="https://github.com/user-attachments/assets/de3535ab-2e30-4c1b-a2d1-da9d3ce87c4d" />

* Finding the bottom 3 improvement areas.

```python
avg_ratings.tail(3)
```
Here is the output:
<img width="1066" height="77" alt="13" src="https://github.com/user-attachments/assets/cf29da0b-70d8-4ab5-8ed3-4046ca228e35" />

## Key Insights
* Overall student satisfaction is generally high across the dataset.
* Teaching quality and subject expertise are clear strengths.
* Assignment difficulty and student support consistently score lower than other parameters.
* Improvements in course structure and support services could significantly raise satisfaction levels.
* Student recommendations are strongly influenced by teaching clarity and subject understanding.

## Key Recommendations

Based on the analysis, the following actions are recommended: 

### 1. Improve Assignment Design 

* Balance difficulty with clarity 
* Provide detailed guidelines and examples 

### 2. Enhance Student Support Systems 

* Introduce more doubt-clearing sessions 
* Improve access to academic assistance 

### 3. Refine Course Structure 

* Organise content in a more progressive learning flow 
* Clearly define learning outcomes for each module 

### 4. Leverage Teaching Strengths 

* Encourage best practices from high-performing instructors 
* Use peer mentoring and teaching workshops

# Conclusion

This project demonstrates how structured student feedback can be transformed into actionable insights using data analysis techniques. By combining rating analysis, sentiment categorisation, and visualisation, the study provides a clear understanding of student satisfaction and highlights areas that require attention.

The findings can support academic decision-making, improve learning experiences, and enhance overall institutional effectiveness.
