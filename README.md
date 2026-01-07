# FUTURE_DS_03
Repository for Task 3 of the Data Science &amp; Analytics Internship at Future Interns

# Student Feedback Analysis

# Overview
## Project Overview

The objective of this project is to analyse student feedback data collected after academic courses and learning experiences, identify satisfaction patterns, and extract actionable insights that can help improve teaching quality and overall student experience.

The analysis focuses on: 
• Understanding rating trends across multiple feedback parameters 
• Measuring overall satisfaction levels 
• Deriving sentiment categories based on quantitative feedback 
• Providing data-driven recommendations for improvement

## Dataset Overview

* Dataset: Student Feedback Ratings (CSV)
Link: https://www.kaggle.com/datasets/ruchi798/student-feedback-survey-responses
* Records Analysed: 1001 student responses

The dataset consists of numerical ratings provided by students on a scale of 1 to 10 for the following aspects: 
• Subject Knowledge 
• Teaching Clarity 
• Use of Presentations 
• Assignment Difficulty 
• Doubt Solving 
• Course Structure 
• Student Support 
• Course Recommendation 

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










