# imports
import math
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# extract data from the housing market data csv file
# Pandas DataFrame manipulation
def read_houses_data(filename):
    df = pd.read_csv(filename)  # Load data from csv file

    # arrange data: rename columns, drop columns and rows
    columns_name = df.iloc[10:11, :6]  # cut specific row contain new names for columns labeling
    df = df.iloc[11:, :6]  # remove unneeded columns and rows
    dict = {}
    for i in range(6):
        dict[df.columns[i]] = columns_name.iloc[0][i]  # create dict for rename function
    df = df.rename(columns=dict).reset_index()
    df = df.drop(columns=["Yearly Average", "index"])  # drop unneeded columns
    df = df.replace(',', '', regex=True)
    return df

# extract data from the stock market data csv file
def read_stock_data(filename):
    df = pd.read_csv(filename)  # Load data from csv file
    dates = []
    for i in range(18):
        # take only sample of each quarter in the year
        dates += [str(2000 + i) + "-03-01", str(2000 + i) + "-06-01",
                  str(2000 + i) + "-09-01", str(2000 + i) + "-12-01"]
    df = df[df['Date'].str.contains("|".join(dates))]
    return df.SP500 # return the series with index tag "SP500"


# Numpy Matrix-array manipulation
def houses_array_manipulation(df):
    arr1 = df.values.astype(np.float64)  # convert String DateFrame to Float 2'd array

    # arrange data: create rows for quarters of each year, convert values of cost from M to K
    # Takes advantage of numpy array efficient manipulation and calculation
    quarters = np.array([0, 0.25, 0.5, 0.75], dtype=np.float16)

    # each iteration creating a (2,4) shape array in which the first row contains the yearly quarters and the second
    # row contain the housing cost
    final = np.array([]) # init empty numpy array
    for i in range(arr1.shape[0]):
        arr2 = (np.ones(shape=(2, 4), dtype=np.float16)) * int(arr1[i][4]) + quarters
        # first 3 rows are in Shekels instead of K-Shekels
        if i < 3:
            arr2[1] = arr1[i][:4] / 1000
        else:
            arr2[1] = arr1[i][:4]
        if i == 0:
            final = arr2.T
        else:
            final = np.concatenate((final, arr2.T), axis=0)  # concatenate current array with previous arrays
    return final.T

# Define investment strategy for 20 years in real-estate and stocks
def investment_strategy(house_price, Hm, Hb, Sm, Sb):
    # mortgage parameters
    Startup_capital = house_price / 3  # start-up capital in K
    loan = 2 * Startup_capital  # first investment = 1/3 and mortgage = 2/3 of investment value
    N = 20 * 12  # number of payment for the return of the mortgage (months)
    interest = 0.025 / 12
    monthly_payment = loan * interest * math.pow((1 + interest), N) / (math.pow((1 + interest), N) - 1)
    mortgage_interest = monthly_payment * N - loan

    # calculate profit in house and stock in the years 2017 - 2037
    # assume monthly rent = monthly payment of the mortgage
    house_profit = np.round(float(Hm * 2037 + Hb - mortgage_interest - Startup_capital), 2)
    stock_profit = np.round(float(Sm * 2037 + Sb - Startup_capital), 2)
    house_return = np.round(100 * (house_profit / Startup_capital), 2)  # percentage
    stock_return = np.round(100 * (stock_profit / Startup_capital), 2)  # percentage

    return house_profit, stock_profit, house_return, stock_return

# Matplotlib visualization tools
def visualization(x_axes, Hy_axes, Sy_axes, Hm, Hb, Sm, Sb):
    fig = plt.figure()  # create a figure to contain all the MATPLOT visualization

    # add dots to graph present each sample of the data
    plt.scatter(x_axes, Hy_axes, alpha=1, color='royalblue', label="Tel_Aviv Housing Price")
    plt.scatter(x_axes, Sy_axes, alpha=1, color='firebrick', label="S&P500 1000 Stocks Price")

    # add Line to graph present the linear regression computed from the AI algorithm
    plt.plot((1999, 2019), (Hm * 1999 + Hb, Hm * 2019 + Hb), color='lightseagreen', linestyle='-', linewidth=3,
             label="Houses-line")
    plt.plot((1999, 2019), (Sm * 1999 + Sb, Sm * 2019 + Sb), color='deeppink', linestyle='-', linewidth=3,
             label="Stock-line")

    # style the graph
    plt.grid()
    plt.xlim(1999, 2019)

    # add text
    plt.title("TLV Housing/S&P500 1000 Stocks Price over Time")
    plt.xlabel("Year Of Date")
    plt.ylabel("1000 Israeli New Shekels")
    plt.legend()

    # plt.show() # Hidden as the figure will present in the PDF
    return fig

# Matplotlib - write text to matplotlib figure
def report(Hp, Sp, Hr, Sr):
    fig = plt.figure()  # create a figure to contain all the MATPLOT visualization

    # create text report
    plt.figtext(0.5, 0.5, "TLV Housing and S&P500 stock prediction report:\n\n"
                          "Profit prediction for investment in housing in TLV\n"
                          "in 2017 after 20 years is: " + str(Hp) +
                "K ₪.\nThe return is: " + str(Hr) +
                "%.\n\n Profit prediction for investment in the S&P500 stock\n"
                "with the same start-up capital,\n"
                "after 20 years is: " + str(Sp) +
                "K ₪.\nThe return is: " + str(Sr) + "%.",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=16,
                )
    return fig

# Matplotlib - create PDF
def print_pdf(fig1, fig2):
    # create PDF and fill it with report figures
    pp = PdfPages('Report.pdf')
    pp.savefig(fig1)
    pp.savefig(fig2)
    pp.close()

# Main method to run software
def main():
    # preparing the data: define predictor and response variable
    tel_aviv = read_houses_data("tel_aviv_2000_2017.csv")
    data_array = houses_array_manipulation(tel_aviv)
    X_year = data_array[0]  # predictor
    Y_house = data_array[1]  # response

    Y_sp500 = read_stock_data("S&P500_data.csv")

    # split data into train and test, set test to 33% of the whole data
    Hx_train, Hx_test, Hy_train, Hy_test = train_test_split(
        X_year, Y_house, test_size=0.33, random_state=42)

    Sx_train, Sx_test, Sy_train, Sy_test = train_test_split(
        X_year, Y_sp500, test_size=0.33, random_state=42)

    # create linear regression model and fit the model on the training set
    Houses_linreg = linear_model.LinearRegression()
    Houses_linreg.fit(Hx_train.reshape(-1, 1), Hy_train)

    Stock_linreg = linear_model.LinearRegression()
    Stock_linreg.fit(Sx_train.reshape(-1, 1), Sy_train)

    Hp, Sp, Hr, Sr = investment_strategy(Y_house[-1],
                                         Houses_linreg.coef_, Houses_linreg.intercept_,
                                         Stock_linreg.coef_, Stock_linreg.intercept_)
    # present in a graph
    fig1 = visualization(X_year, Y_house, Y_sp500,
                         Houses_linreg.coef_, Houses_linreg.intercept_,
                         Stock_linreg.coef_, Stock_linreg.intercept_)

    # create the prediction report data
    fig2 = report(Hp, Sp, Hr, Sr)

    # summarize all in a pdf name Report.pdf
    print_pdf(fig1, fig2)

    print("Report Ready")


# program init
main()
