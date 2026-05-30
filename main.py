import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv.xls")

plt.figure(figsize=(8,5))
sns.boxplot(x="Churn", y="tenure", data=df)

plt.title("Tenure vs Churn")

plt.savefig("images/tenure_vs_churn.png")

plt.show()