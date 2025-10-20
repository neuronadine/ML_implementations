import numpy as np

class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train.astype(int)
        self.x_val = x_val
        self.y_val = y_val.astype(int)

    def hard_parzen(self, h):
        hard_parzen = HardParzen(h)
        hard_parzen.fit(self.x_train, self.y_train)
        y_pred = hard_parzen.predict(self.x_val)
        n_error=np.sum(y_pred != self.y_val)
        return n_error / len(self.y_val)


    def soft_parzen(self, sigma):
        soft_parzen = SoftRBFParzen(sigma)
        soft_parzen.fit(self.x_train, self.y_train)
        y_pred = soft_parzen.predict(self.x_val)
        n_error=np.sum(y_pred != self.y_val)
        return n_error / len(self.y_val)


def get_test_errors(iris):
    train, validation, test = split_dataset(iris)

    # Extract features and labels for each set
    x_train = train[:,:4]
    y_train = train[:,4].astype(int)
    x_val = validation[:,:4]
    y_val = validation[:,4].astype(int)
    x_test = test[:,:4]
    y_test = test[:,4].astype(int)

    # initiate ErrorRate object for hyperparameter tuning
    err_val = ErrorRate(x_train, y_train, x_val, y_val)

    # Values of h and sigma as specified in the assignment
    h_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    sigma_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]

    # Find best h
    best_err_rate = np.inf
    best_h = None
    for h in h_values:
        err_rate = err_val.hard_parzen(h)
        if err_rate < best_err_rate :
            best_err_rate = err_rate
            best_h = h
    
    # Find best sigma
    best_err_rate = np.inf
    best_sigma = None
    for sigma in sigma_values:
        err_rate = err_val.soft_parzen(sigma)
        if err_rate < best_err_rate :
            best_err_rate = err_rate
            best_sigma = sigma

    # Predict using the best_h and best_sigma
    err_test = ErrorRate(x_train, y_train, x_test, y_test)
    err_test_hard = err_test.hard_parzen(best_h)
    err_test_soft = err_test.soft_parzen(best_sigma)

    return np.array([err_test_hard, err_test_soft])


