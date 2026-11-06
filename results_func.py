import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

brainFrame = pd.read_csv('brainsize.txt', delimiter='\t', na_values='NA')

menDf = brainFrame[brainFrame['Gender'] == 'Male']
womenDf = brainFrame[brainFrame['Gender'] == 'Female']

# для женщин
plt.figure(figsize=(7, 5))
sns.heatmap(womenDf.corr(numeric_only=True, method='pearson'), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Тепловая карта корреляций (Женщины)")
plt.show()

# для мужчин
plt.figure(figsize=(7, 5))
sns.heatmap(menDf.corr(numeric_only=True, method='pearson'), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Тепловая карта корреляций (Мужчины)")
plt.show()
