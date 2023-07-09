from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    equal=0
    for i in range(len(y)):
        if(y_hat.iloc[i]==y.iloc[i]):
            equal+=1;
    return (equal/len(y))*100


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size==y.size
    count=0
    for i in range(len(y)):
        if(y_hat.iloc[i]==y.iloc[i] and y_hat.iloc[i]==cls):
            count+=1
    if count==0:
        return 0
    return count/len(y_hat[y_hat==cls])


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size==y.size
    count=0
    for i in range(len(y)):
        if(y_hat.iloc[i]==y.iloc[i] and y_hat.iloc[i]==cls):
            count+=1
    return count/(len(y[y==cls]))


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size==y.size
    l=pd.Series()
    for i in range(len(y)):
        l.at[i]=((y.iloc[i]-y_hat.iloc[i])**2)
    
    return l.mean()**0.5


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size==y.size
    l=pd.Series()
    for i in range(len(y)):
        l.at[i]=(np.abs(y.iloc[i]-y_hat.iloc[i]))
    
    return l.mean()
