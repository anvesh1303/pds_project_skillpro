import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

resumeDataSet = pd.read_csv('/Users/anveshkumar/Desktop/pds_resume_project/data/resume_dataset.csv', encoding='utf-8')

le = LabelEncoder()
resumeDataSet['Category'] = le.fit_transform(resumeDataSet['Category'])

resumeDataSet['Category_names'] = le.inverse_transform(resumeDataSet['Category'])

plt.figure(figsize=(12, 6))
sns.countplot(data=resumeDataSet, x='Category_names', order=resumeDataSet['Category_names'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Distribution of Resume Categories')
plt.show()


