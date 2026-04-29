import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df):
    """
    Return a list of matplotlib figures for EDA.
    """
    figs = []
    # Churn distribution
    fig1, ax1 = plt.subplots()
    if 'Churn' in df.columns:
        sns.countplot(x='Churn', data=df, ax=ax1)
        ax1.set_title("Churn Distribution")
        figs.append(fig1)
    # Tenure histogram
    if 'tenure' in df.columns:
        fig2, ax2 = plt.subplots()
        sns.histplot(df['tenure'], kde=True, ax=ax2)
        ax2.set_title("Tenure Distribution")
        figs.append(fig2)
    # Monthly charges vs churn
    if 'MonthlyCharges' in df.columns and 'Churn' in df.columns:
        fig3, ax3 = plt.subplots()
        sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax3)
        figs.append(fig3)
    return figs

if __name__ == "__main__":
    df = pd.read_csv("../data/telco_data.csv")
    explore_data(df)
    plt.show()