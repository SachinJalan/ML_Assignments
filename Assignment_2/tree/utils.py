import pandas as pd
import numpy as np

def entropy2(Y: pd.Series, weight:pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    entrp=0;
    totsum=sum(weight)
    for i in Y.cat.categories:
        x=(sum(weight[Y==i])/totsum)

        if(x!=0):
            entrp+=(-x*np.log2(x))
        else:
            entrp+=0
    return entrp
def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    ser = Y.value_counts()
    px = ser / (ser.sum())
    px2 = px.apply(lambda x: -x * np.log2(x))
    return px2.sum()
def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    ser = Y.value_counts()
    px = ser / ser.sum()
    px2 = px.apply(lambda x: x**2)
    return 1 - (px2.sum())

def information_gain_RealAda(Y: pd.Series, attr: pd.Series,weights: pd.Series,flag=1):
    sort_Y = Y.iloc[attr.argsort()]
    attr2 = attr.iloc[attr.argsort()]
    sort_W = weights.iloc[attr.argsort()]
    maxIG = -2
    val = -1
    if flag == 1:
        entr = entropy2(Y,weights)
    for i in range(len(sort_Y) - 1):
        if sort_Y.iloc[i] != sort_Y.iloc[i + 1]:
            if flag == 1:
                tempIG = (
                    entr
                    - ((i / len(sort_Y)) * entropy2(sort_Y.iloc[0 : i + 1],sort_W.iloc[0 : i + 1]))
                    - (
                        (len(sort_Y) - i)
                        / (len(sort_Y))
                        * entropy2(sort_Y.iloc[i + 1 :],sort_W.iloc[i + 1 :])
                    )
                )
            else:
                tempIG = (
                    entr
                    - ((i / len(sort_Y)) * gini_index(sort_Y.iloc[0 : i + 1]))
                    - (
                        (len(sort_Y) - i)
                        / (len(sort_Y))
                        * gini_index(sort_Y.iloc[i + 1 :])
                    )
                )
            if tempIG > maxIG:
                maxIG = tempIG
                val = (attr2.iloc[i] + attr2.iloc[i + 1]) / 2
    return (maxIG, val)

def information_gain(Y: pd.Series, attr: pd.Series, flag=1) -> float:
    """
    Function to calculate the information gain
    """
    df_out = pd.crosstab(attr, Y)
    df_out2 = pd.crosstab(attr, Y, normalize="index")
    if flag == 1:
        df_out3 = df_out2.apply(lambda x: -x * np.log2(x)).fillna(0)
    else:
        df_out3 = df_out2.apply(lambda x: -x * (1 - x)).fillna(0)
    df_out4 = df_out.sum(axis=1)
    df_out5 = df_out4 / len(Y)
    df_entropy = df_out3.sum(axis=1)
    df_infogain = df_out5 * df_entropy
    info_gain = df_infogain.sum()
    if flag == 1:
        entropy_Y = entropy(Y)
    else:
        entropy_Y = gini_index(Y)
    return entropy_Y - info_gain


def information_gain_DIRO(Y: pd.Series, attr: pd.Series) -> float:
    sumentr = 0
    lenn = len(Y)
    for i in attr.unique():
        xx = Y[attr == i]
        sumentr += (len(Y[attr == i]) / (lenn)) * (xx.var())
    VAR = Y.var()
    return VAR - sumentr


def mse(Y: pd.Series) -> float:
    return ((Y - Y.mean()) ** 2).mean()


def information_gain_Real2(Y: pd.Series, attr: pd.Series):
    sort_Y = Y.iloc[attr.argsort()]
    attr2 = attr.iloc[attr.argsort()]
    maxIG = -2
    val = -1
    rmseT = mse(Y)
    for i in range(len(sort_Y) - 1):
        IGTemp = rmseT - (mse(sort_Y.iloc[0 : i + 1]) + mse(sort_Y.iloc[i + 1 :]))
        if IGTemp > maxIG:
            maxIG = IGTemp
            val = (attr2.iloc[i] + attr2.iloc[i + 1]) / 2
    return (maxIG, val)



def information_gain_Real(Y: pd.Series, attr: pd.Series, flag=1):
    sort_Y = Y.iloc[attr.argsort()]
    attr2 = attr.iloc[attr.argsort()]
    maxIG = -2
    val = -1
    if flag == 1:
        entr = entropy(Y)
    else:
        entr = gini_index(Y)
    for i in range(len(sort_Y) - 1):
        if sort_Y.iloc[i] != sort_Y.iloc[i + 1]:
            if flag == 1:
                tempIG = (
                    entr
                    - ((i / len(sort_Y)) * entropy(sort_Y.iloc[0 : i + 1]))
                    - (
                        (len(sort_Y) - i)
                        / (len(sort_Y))
                        * entropy(sort_Y.iloc[i + 1 :])
                    )
                )
            else:
                tempIG = (
                    entr
                    - ((i / len(sort_Y)) * gini_index(sort_Y.iloc[0 : i + 1]))
                    - (
                        (len(sort_Y) - i)
                        / (len(sort_Y))
                        * gini_index(sort_Y.iloc[i + 1 :])
                    )
                )
            if tempIG > maxIG:
                maxIG = tempIG
                val = (attr2.iloc[i] + attr2.iloc[i + 1]) / 2
    return (maxIG, val)

