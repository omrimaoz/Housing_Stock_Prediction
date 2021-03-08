# imports
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Pandas DataFrame manipulation
def read_data():
    df = pd.read_csv("tel_aviv_2000_2017.csv") # Load data from csv file

    # arrange data: rename columns, drop columns and rows
    columns_name = df.iloc[10:11, :6] # cut specific row contain new names for columns labeling
    df = df.iloc[11:, :6] # remove unneeded columns and rows
    dict = {}
    for i in range(6):
        dict[df.columns[i]] = columns_name.iloc[0][i] #create dict for rename function
    df = df.rename(columns=dict).reset_index()
    df = df.drop(columns=["Yearly Average", "index"]) # drop unneeded columns
    df = df.replace(',', '', regex=True)
    return df

# Numpy Matrix-array manipulation
def array_manipulation(df):
    arr1 = df.values.astype(np.float64) # convert String DateFrame to Float 2'd array
    # arrange data: create rows for quarters of each year, convert values of cost from M to K
    # Takes advantage of numpy array efficient manipulation and calculation
    quarters = np.array([0,0.25,0.5,0.75], dtype=np.float16)

    # each iteration creating a (2,4) shape array in which the first row contains the yearly quarters and the second
    # row contain the housing cost
    for i in range(arr1.shape[0]):
        arr2 = (np.ones(shape=(2, 4), dtype=np.float16)) * int(arr1[i][4]) + quarters
        if i < 3:
            arr2[1] = arr1[i][:4]/1000
        else:
            arr2[1] = arr1[i][:4]
        if i == 0:
            final = arr2.T
        else:
            final = np.concatenate((final, arr2.T), axis=0)  # concatenate current array with previous arrays
    return final.T

def visualization2d(y_axes,x_axes):
    fig = plt.figure()
    # Colors = plt.cm.viridis(np.linspace(0, 1, K))
    # for i in range(K):
    #     if lst[i].size != 0:
    #         newMatrix = dataMatrix[lst[i]].T
    #         x_axes = newMatrix[0]
    #         y_axes = newMatrix[1]
    #         plt.scatter(x_axes, y_axes, c=[(Colors[i][0],Colors[i][1],Colors[i][2])] ,label = "cluster " +str(i), alpha = 1)
    #     else:
    #         print("In " + alg_name + " ,cluster number " + str(i) + " is empty")
    plt.scatter(y_axes, x_axes, alpha = 1)
    plt.grid()
    plt.title("Housing Price X Time")
    plt.legend() #TODO: remove legend and label

    plt.show()
    return fig

def main():
    # preparing the data: define predictor and response variable
    tel_aviv = read_data()
    data_array = array_manipulation(tel_aviv)
    X_year = data_array[0] # predictor
    Y_cost = data_array[1] # response
    # visualization2d(X_year,Y_cost)

    # split data into train and test, set test to 33% of the whole data
    x_train, x_test, y_train, y_test = train_test_split(
        X_year, Y_cost, test_size = 0.33, random_state=42)

    # create linear regression model and fit the model on the training set
    linreg = linear_model.LinearRegression()
    linreg.fit(x_train, y_train)



    print("done")
main()
