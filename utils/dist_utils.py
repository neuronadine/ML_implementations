import numpy as np

# Minkowski
def minkowski(x,y,p=2):
  return np.sum(np.abs(x-y)**p)**(1/p)

def minkowski_mat(X,y,p=2):
  return np.sum(np.abs(X-y)**p, axis=1)**(1/p)

# Manhattan
def manhattanL1(X, y):
    return np.sum(np.abs(X - y), axis=1)

# Euclidean

# Tanimoto

def random_projections(X, A):
    return (1 / np.sqrt(2)) * np.dot(X, A)

def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)


def feature_means(self, iris):
    return iris[:, :4].mean(axis=0)

def empirical_covariance(self, iris):
    return np.cov(iris[:, :4], rowvar=False)

def feature_means_class_1(self, iris):
    iris_class_1 = iris[iris[:, 4] == 1]
    return self.feature_means(iris_class_1)

def empirical_covariance_class_1(self, iris):
    iris_class_1 = iris[iris[:, 4] == 1]
    return self.empirical_covariance(iris_class_1)


def split_dataset(iris):
    n = iris.shape[0]
    train = iris[(np.arange(n)%5 <= 2)]
    validation = iris[(np.arange(n)%5 == 3)]
    test = iris[(np.arange(n)%5 == 4)]
    return train, validation, test


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


