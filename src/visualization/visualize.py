import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
df = pd.read_csv(r"C:\Users\nihall\Desktop\MLops-Diabities\source_data\pima_indian.csv")

# Create a bar plot with pandas
ax = df.plot(x='diabetes', y='age', kind='bar')
ax.set_title('Diabetes Outcome by Age')
ax.set_xlabel('Diabetes Outcome')
ax.set_ylabel('Age')

# Ensure the directory exists and save the plot
save_dir = r"../src/visualization"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'age_vs_diabetes_bar_plot.png'))

# Create a bar plot with seaborn
plt.figure()  # Create a new figure to avoid overlapping plots
ds = sns.barplot(x=df['diabetes'])
ds.set_title("Diabetes Outcome Bar Plot")

# Save the seaborn plot
plt.savefig(os.path.join(save_dir, 'outcome.png'))

# Optionally, display the plot (not required if only saving)
plt.show()
