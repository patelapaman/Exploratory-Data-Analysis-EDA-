import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("iris.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())

numeric = df.select_dtypes(include="number")

print("\nMean:\n", numeric.mean())
print("\nMedian:\n", numeric.median())
print("\nMode:\n", numeric.mode().iloc[0])
print("\nStandard Deviation:\n", numeric.std())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

os.makedirs("plots", exist_ok=True)

numeric.hist(figsize=(10, 8))
plt.suptitle("Distribution of Iris Features")
plt.tight_layout()
plt.savefig("plots/feature_distributions.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=numeric)
plt.title("Boxplots of Iris Features")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("plots/boxplots.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="sepal_length", y="petal_length", hue="species")
plt.title("Sepal Length vs Petal Length")
plt.tight_layout()
plt.savefig("plots/sepal_vs_petal.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=300)
plt.show()

print("\nEDA completed successfully!")
