'''
These are basic steps, and the specific preprocessing steps may vary based on the nature of your data and the
requirements of your analysis or model. Always explore and understand your data before applying preprocessing
techniques.
'''

from pandas import DataFrame, read_csv
from numpy import log, nonzero
from numpy import sqrt
from numpy import where
from seaborn import boxplot
from typing import List
from matplotlib.pylab import show
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from scipy.stats.mstats import winsorize

'''*********************************************************************************************************************
Identifying Missing Values:
Before handling missing values, it's crucial to identify where they exist in your dataset. You can use libraries like 
Pandas to check for missing values:
*********************************************************************************************************************'''
def find_missing():
    for i in range(8):
            for j in range(2):
                print(str(i) + "_" + str(j) + ":")
                PATH = "./compdata/bdhsc_2024/stage1_labeled/" + str(i) + "_" + str(j) + ".csv"
                df = read_csv(PATH)

                missing = df.isnull().sum()
                print(nonzero(missing.iloc[:].values))

'''*********************************************************************************************************************
Splitting Data:
If you're building a machine learning model, split the dataset into training and testing sets to evaluate the model's 
performance.
*********************************************************************************************************************'''
def xy_split(data: DataFrame, dependent_col: int) -> DataFrame:
    X = data.drop(data.columns[dependent_col], axis=1)
    y = data[:, dependent_col]
    return (X, y)

def tt_split(X: DataFrame, y: DataFrame, test_size=0.2) -> (DataFrame, DataFrame, DataFrame, DataFrame):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    # The 'test_size' parameter determines the proportion of the dataset to include in the test split.
    # 'random_state' ensures reproducibility by fixing the random seed.
    # Now you can use X_train and y_train for training your machine learning model,
    # and X_test and y_test for evaluating its performance.

    return (X_train, X_test, y_train, y_test)


''' ********************************************************************************************************************
Imbalanced data:
refers to a situation where the distribution of classes in a dataset is not uniform, meaning that one class has 
significantly fewer instances than the others. Dealing with imbalanced data is crucial, as many machine learning 
algorithms assume a balanced class distribution. 
*********************************************************************************************************************'''

'''
Random Oversampling: Duplicate instances of the minority class.
Random Undersampling: Remove instances of the majority class
SMOTE (Synthetic Minority Over-sampling Technique): Generate synthetic samples for the minority class.
NearMiss: Remove instances from the majority class based on distance.
'''
def rnd_sampling(X: DataFrame, y: DataFrame, sample_type="ros") -> (DataFrame, DataFrame):
    if sample_type == "ros":
        sampler = RandomOverSampler(random_state=42)
    elif sample_type == "rus":
        sampler = RandomUnderSampler(random_state=42)
    elif sample_type == "smote":
        sampler = SMOTE(random_state=42)
    elif sample_type == "nm":
        sampler = NearMiss(version=1)
    else:
        sampler = RandomOverSampler(random_state=42)
    return sampler.fit_resample(X, y)