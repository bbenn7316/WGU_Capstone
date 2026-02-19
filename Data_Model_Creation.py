from Data_Load_Preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error

def separate(df):
    genrey = df["Positive"]
    genrex = df[["Metacritic score", "Recommendations", "Average playtime forever", "Peak CCU"]]

    genrex_train, genrex_test, genrey_train, genrey_test = train_test_split(genrex, genrey, test_size = 0.2, random_state = 100)

    return  genrex_train, genrex_test, genrey_train, genrey_test

def create_MLM(genrex_train, genrex_test, genrey_train, genrey_test):

    model = LinearRegression()
    model.fit(genrex_train, genrey_train)

    model_train_pred = model.predict(genrex_train)
    model_test_pred = model.predict(genrex_test)

    return model, model_train_pred, model_test_pred

def RMSE_R2(ytrain, ytest, model_train_pred, model_test_pred):

    mse_train = mean_squared_error(ytrain, model_train_pred)
    r2_train = r2_score(ytrain, model_train_pred)

    mse_test = mean_squared_error(ytest, model_test_pred)
    r2_test = r2_score(ytest, model_test_pred)

    mae = mean_absolute_error(ytest, model_test_pred)

    return mse_train, r2_train, mse_test, r2_test, mae