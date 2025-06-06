import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
pd.set_option('display.max.rows',130)
pd.set_option('display.max.columns',130)
pd.set_option('float_format', '{:.2f}'.format)


def bi_cat_countplot_sub(df, column, hue_column, ax, size=20):
    unique_hue_values = df[hue_column].unique()

    pltname = f'Normalised distribution of values by category: {column}'
    proportions = df.groupby([hue_column, column]).size().unstack(hue_column, fill_value=0)
    proportions = (proportions * 100 / proportions.sum()).round(2)
    plot = proportions.sort_index().sort_values(
        by=unique_hue_values[0], ascending=False
    ).plot.bar(ax=ax, title=pltname)

    # анотація значень в барплоті
    for container in plot.containers:
        plot.bar_label(container, fmt='%1.1f%%')

    # Sort x-labels by alphabet
    ax.set_xticklabels(sorted(ax.get_xticklabels(), key=lambda label: label.get_text()))

def uni_cat_target_compare_sub(df, column, ax):
    bi_cat_countplot_sub(df, column, hue_column='y', ax=ax)


def create_subplots(df, columns):
    # Adjust the number of rows and columns based on the number of columns
    n_cols = 2  # Number of columns in the subplot grid
    n_rows = (len(columns) + n_cols - 1) // n_cols  # Calculate rows dynamically
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        uni_cat_target_compare_sub(df, col, ax=axes[i])
    
    # Hide any unused subplots
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def bi_cat_countplot(df, column, hue_column, size=20):
    unique_hue_values = df[hue_column].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(size, 6)

    pltname = f'Normalised distribution of values by category: {column}'
    proportions = df.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions * 100).round(2)
    ax = proportions.unstack(hue_column).sort_index().sort_values(
        by=unique_hue_values[0], ascending=False
    ).plot.bar(ax=axes[0], title=pltname)

    # анотація значень в барплоті
    for container in ax.containers:
        ax.bar_label(container, fmt='%1.1f%%')

    pltname = f'The amount of data per category: {column}'
    counts = df.groupby(hue_column)[column].value_counts()
    ax = counts.unstack(hue_column).sort_index().sort_values(
        by=unique_hue_values[0], ascending=False
    ).plot.bar(ax=axes[1], title=pltname)

    for container in ax.containers:
        ax.bar_label(container)

    # Sort x-labels by alphabet
    for ax in axes:
        ax.set_xticklabels(sorted(ax.get_xticklabels(), key=lambda label: label.get_text()))

def uni_cat_target_compare(df, column):
    bi_cat_countplot(df, column, hue_column='y' )


def bi_countplot_target(df0, df1, column, hue_column, size = 20):
  pltname = 'The client has taken out a term deposit'
  print(pltname.upper())
  bi_cat_countplot(df1, column, hue_column, size)
  plt.show()

  pltname = 'The client has NOT taken out a term deposit'
  print(pltname.upper())
  bi_cat_countplot(df0, column, hue_column, size)
  plt.show()

import warnings

def dist_box(dataset, column):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")

      plt.figure(figsize=(16,6))

      plt.subplot(1,2,1)
      sns.distplot(dataset[column], color = 'purple')
      pltname = 'Distribution for ' + column
      plt.ticklabel_format(style='plain', axis='x')
      plt.title(pltname)

      plt.subplot(1,2,2)
      red_diamond = dict(markerfacecolor='r', marker='D')
      sns.boxplot(y = column, data = dataset, flierprops = red_diamond)
      pltname = 'Boxplot for ' + column
      plt.title(pltname)

      plt.show()

def outlier_range(dataset, column):
    Q1 = dataset[column].quantile(0.25)
    Q3 = dataset[column].quantile(0.75)
    IQR = Q3 - Q1
    Min_value = (Q1 - 1.5 * IQR)
    Max_value = (Q3 + 1.5 * IQR)
    print(f"The maximum value after which there are outliers for {column}: {Max_value}")
    return Max_value

def outlier_analisys(df0, df1, column):
    
    max_v0 = outlier_range(df0, column)
    max_v1 = outlier_range(df1, column)

    if max_v0 > df0[column].max():
        print('No Outliers in df0')
    if max_v1 > df1[column].max():
        print('No Outliers in df1')

    kde_full(df0, df1, column)
    kde_no_outliers(df0, df1, max_v0, max_v1, column)    
    return max_v0, max_v1  
  
def kde_no_outliers(df0, df1, Max_value0, Max_value1, column):
  plt.figure(figsize = (14,6))
  sns.kdeplot(df1[df1[column] <= Max_value1][column],label = 'Got term deposit')
  sns.kdeplot(df0[df0[column] <= Max_value0][column],label = 'has NOT got term deposit')
  plt.ticklabel_format(style='plain', axis='x')
  plt.xticks(rotation = 45)
  plt.legend()
  plt.title(f'KDE plot for {column} W/O Outliers')
  plt.show()

def kde_full(df0, df1, column):
  plt.figure(figsize = (14,6))
  sns.kdeplot(df1[column],label = 'Got term deposit')
  sns.kdeplot(df0[column],label = 'has NOT got term deposit')
  plt.ticklabel_format(style='plain', axis='x')
  plt.xticks(rotation = 45)
  plt.title(f'KDE plot for {column} with Outliers')
  plt.legend()
  plt.show()     

def draw_boxplot(df, categorical, continuous, max_continuous, title, hue_column, subplot_position):
    """
    Малює блок-діаграму для заданого DataFrame, категоріальної та неперервної змінної.
    """
    plt.subplot(1, 2, subplot_position)
    plt.title(title)
    red_diamond = dict(markerfacecolor='r', marker='D')
    sns.boxplot(x=categorical,
                y=df[df[continuous] < max_continuous][continuous],
                data=df,
                flierprops=red_diamond,
                order=sorted(df[categorical].unique(), reverse=True),
                hue=hue_column, hue_order=sorted(df[hue_column].unique(), reverse=True))
    plt.ticklabel_format(style='plain', axis='y')
    plt.xticks(rotation=90)

def bi_boxplot(df1, df0, categorical, continuous, max_continuous1, max_continuous0, hue_column):
    """
    Створює паралельні блок-діаграми для двох груп, визначених у наборі даних, на основі
    категоріальної та неперервної змінної, виділяючи відмінності за допомогою відтінків.
    """
    plt.figure(figsize=(16, 10))

    # Графік для першо групи "Труднощі з платежами" (Payment Difficulties)
    draw_boxplot(df1, categorical, continuous, max_continuous1, 'Got term deposit', hue_column, 1)

    # Графік для другої групи "Вчасні оплати" (On-Time Payments)
    draw_boxplot(df0, categorical, continuous, max_continuous0, 'has NOT got term deposit', hue_column, 2)

    plt.tight_layout(pad=4)
    plt.show()  

def numeric_vs_categorical_analysis(df0, df1, column_1, column_2, column_3, target):
  max_value1_column_1 = outlier_range(df1, column_1)
  max_value0_column_1 = outlier_range(df0, column_1)

  bi_boxplot(column_2, column_1, max_value1_column_1, max_value0_column_1, column_3, hue_column = target)
