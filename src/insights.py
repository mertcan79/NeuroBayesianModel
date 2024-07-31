from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def compare_performance(model, data, target_variable):
    X = data.drop(columns=[target_variable])
    y = data[target_variable]
    
    # Your model's performance
    y_pred_model = model.predict(data)
    mse_model = mean_squared_error(y, y_pred_model)
    r2_model = r2_score(y, y_pred_model)
    
    # Linear regression performance
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred_lr = lr.predict(X)
    mse_lr = mean_squared_error(y, y_pred_lr)
    r2_lr = r2_score(y, y_pred_lr)
    
    return {
        "NeuroBayesianModel": {"MSE": mse_model, "R2": r2_model},
        "Linear Regression": {"MSE": mse_lr, "R2": r2_lr}
    }