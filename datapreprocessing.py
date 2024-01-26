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
Filling Missing Values:
You can fill missing values with a specific value (e.g., mean, median, or mode) using the fillna() method:

Interpolation:
Interpolation is useful when dealing with time series data. It fills missing values based on the existing values in the 
series:

Imputation with Machine Learning Models:
You can use machine learning models to predict missing values based on other features. Libraries like Scikit-Learn 
provide tools for this purpose:

Handling Categorical Missing Values:
For categorical data, you might want to replace missing values with the mode (most frequent category) or a special 
category like "Unknown":
*********************************************************************************************************************'''
def fill_missing(data: DataFrame, fill_type: str="value", value: int=0) -> DataFrame:
    if fill_type == "value":
        data = data.fillna(value=value)
    elif fill_type == "interpolate":
        data = data.interpolate()
    elif fill_type == "mean":
        for col in data.keys():
            data[col] = data[col].fillna(data[col].mean())
    elif fill_type == "median":
        for col in data.keys():
            data[col] = data[col].fillna(data[col].median())
    elif fill_type == "mode":
        for col in data.keys():
            data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data = data.fillna(value=value)
    return data

''' *********************************************************************************************************************
Handling Duplicates:
Check for and remove any duplicate rows in the dataset.
*********************************************************************************************************************'''

'''
Counting Duplicates:
To count the occurrences of each value and identify duplicates, you can use value_counts():
'''
def count_duplicates(data: DataFrame, col: str) -> None:
    # Count occurrences of each value
    print(data[col].value_counts())

'''
Identifying Duplicate Rows:
Use the duplicated() method to identify duplicate rows in a DataFrame.
'''
def find_duplicates(data: DataFrame) -> None:
    print(data[data.duplicated()])

'''
Removing Duplicate Rows:
Use the drop_duplicates() method to remove duplicate rows from your DataFrame.

Keeping the First/Last Occurrence:
If you want to keep the first occurrence of each duplicate and remove the subsequent ones, use the keep parameter 
in drop_duplicates():
'''
def rm_duplicates(data: DataFrame, subset=None, keep="first") -> DataFrame:
    # Remove duplicate rows
    return data.drop_duplicates(subset=subset, keep=keep)

'''
Marking Duplicates:
You can create a new column to mark duplicates, making it easier to filter or analyze them:
'''
def mark_duplicates(data: DataFrame) -> DataFrame:
    # Create a new column to mark duplicates
    data['is_duplicate'] = data.duplicated()
    return data


''' *********************************************************************************************************************
Handling Outliers:
Identify and handle outliers in the data. This may involve removing outliers or transforming the data using techniques 
like log transformation.
*********************************************************************************************************************'''

'''Identifying Outliers:
Use statistical methods or visualization techniques to identify outliers. Common methods include box plots, scatter 
plots, or statistical measures like Z-scores.'''
def find_outliers(data: DataFrame, cols: List[str]) -> None:
    for col in cols:
        # Visualize outliers with a box plot
        boxplot(x=data[col])
        show()

'''Removing Outliers:
You can remove outliers from your dataset. One approach is to use the Interquartile Range (IQR) to filter out values 
beyond a certain range.'''
def rm_outliers(data: DataFrame, cols: List[str]):
    for col in cols:
        # Calculate IQR
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        # Filter outliers
        data = data[(data[col] >= Q1 - 1.5 * IQR) & (data[col] <= Q3 + 1.5 * IQR)]
    return data

'''Transforming Data:
Apply transformations to make the data more normally distributed. Common transformations include logarithmic, square 
root, or Box-Cox transformations.'''

def transform(data: DataFrame, cols: List[str], transform_type: str="log") -> DataFrame:
    if transform_type == "log":
        data[cols] = log(data[cols] + 1)
    elif transform_type == "sqrt":
        data[cols] = sqrt(data[cols])
    else:
        data[cols] = log(data[cols] + 1)
    return data

'''Winsorizing:
Winsorizing involves replacing extreme values with less extreme values. You can cap or clip the values at a certain 
percentile.'''
def wins(data: DataFrame, cols: List[str], limits=[0.05, 0.05]) -> DataFrame:
    data[cols] = winsorize(data[cols].to_numpy(), limits=limits)
    return data

'''Imputing Values:
If removing outliers is not an option, you can impute values. Replace extreme values with a more reasonable estimate.
'''
def impute(data: DataFrame, cols: List[str], impute_type="median", lower_limit=0, upper_limit=1) -> DataFrame:
# Impute outliers with a specific value (e.g., mean or median)
    for col in cols:
        if impute_type == "median":
            data[col] = where(
                (data[col] < lower_limit) | (data[col] > upper_limit),
                data[col].median(),
                data[col]
            )
        elif impute_type == "mean":
            data[col] = where(
                (data[col] < lower_limit) | (data[col] > upper_limit),
                data[col].mean(),
                data[col]
            )
        elif impute_type == "mode":
            data[col] = where(
                (data[col] < lower_limit) | (data[col] > upper_limit),
                data[col].mode()[0],
                data[col]
            )
        else:
            data[col] = where(
                (data[col] < lower_limit) | (data[col] > upper_limit),
                data[col].median(),
                data[col]
            )
    return data

''' *********************************************************************************************************************
Normalizing/Scaling:
Normalize or scale numerical features to bring them to a similar scale. 
Common methods include min-max scaling or standardization.
*********************************************************************************************************************'''

'''
Scaling
'''
def scale(data: DataFrame, cols: List[str], scale_type: str="std") -> DataFrame:
    if scale_type == "std":
        scaler = StandardScaler()
    elif scale_type == "minmax":
        scaler = MinMaxScaler()
    else:
        print("Unknown scaling type, performing standard scale.")
    data[cols] = scaler.fit_transform(data[cols])
    return data


'''*********************************************************************************************************************
Splitting Data:
If you're building a machine learning model, split the dataset into training and testing sets to evaluate the model's 
performance.
*********************************************************************************************************************'''
def xy_split(data: DataFrame, dependent_cols: List[str]) -> DataFrame:
    X = data.drop(dependent_cols, axis=1)
    y = data[dependent_cols]
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