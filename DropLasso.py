import numpy as np
from scipy import stats
from sklearn import metrics


def main():
    n = 400
    d = 100

    # simulating data
    mu = np.zeros(d)
    correlation_matrix = np.diag(np.ones(d)) + np.outer(np.ones(d), np.ones(d))

    # sampling from multivariate normal distribution
    norm_samples = np.random.multivariate_normal(mu, correlation_matrix, size=n)

    # getting uniform marginals in [0,1]
    uniform_marginals = stats.norm.cdf(norm_samples)

    # transforming the marginals into integers from poisson distribution
    pois = stats.poisson(mu=1)
    X = pois.ppf(uniform_marginals)

    # normalizing the data
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms

    # true model
    w = np.array([0.05] * 10 + [0] * 90)

    grad = mse_grad             # gradient of the loss function
    w0 = np.random.normal(0, 1, d)  # initial weights
    lr = 0.002  # learning rate
    epochs = 10

    num_repeats = 10

    models = {'Elastic Net': {'lambda': True, 'p': False},
              'Dropout': {'lambda': False, 'p': True},
              'DropLasso': {'lambda': True, 'p': True}}

    for q in [0, 0.8, 0.6, 0.4]:
        print(f'q = {q}')
        # introducing noise to the model
        if q > 0:
            delta = np.random.binomial(1, q, d)
            X_noise = X * delta
        else:
            X_noise = X

        # calculating true responses
        probs = 1 / (1 + np.exp(-X_noise @ w))
        y = np.random.binomial(1, probs, n)

        for model_name, params in models.items():
            # model parameters (1 - dropout)
            if params['p']:
                p = 0.5
            else:
                p = 0

            scores = []
            for i in range(num_repeats):
                # model parameters (lambda)
                if params['lambda']:
                    l = i
                else:
                    l = 0

                temp_scores = []
                for j in range(10):
                    # estimating parameters
                    droplasso_estimator = DropLasso(X=X_noise, y=y, grad=grad, w0=w0, lr=lr, epochs=epochs, l=l, p=p)

                    # predicting over the data
                    pred_probs = 1 / (1 + np.exp(-X_noise @ droplasso_estimator))
                    y_pred = np.random.binomial(1, pred_probs, n)

                    # calculating score
                    auc_score = metrics.roc_auc_score(y, y_pred)
                    temp_scores.append(auc_score)

                score = np.mean(temp_scores)
                scores.append(score)

            score = np.mean(scores)
            print(f'\tScore for {model_name}: {round(score, 3)}')



def DropLasso(X, y, grad, w0, lr, epochs, l, p):
    """

    :param X: Training features, individual samples x_i in rows
    :param y: Training labels
    :param grad: gradient of loss function
    :param w0: initial weights
    :param lr: learning rate
    :param epochs: number of passes
    :param l: lambda hyperparamer
    :param p: bernoulli chance of success
    :return w: estimator
    """
    n = X.shape[0]
    d = X.shape[1]
    w = w0
    t = 0
    for i in range(epochs):
        pi = np.random.permutation(n)
        for index in pi:
            t += 1
            lr = lr/(1 + lr*l*t)
            if p > 0:
                delta = np.random.binomial(1, p, d)
                z = (1 / p) * (delta * X[index])
            else:
                z = X[index]
            w = soft_thresholding(x=(w - lr * grad(w, z, y[index])), arg=(lr * l))
    return w


# gradient of the square loss
def mse_grad(w, x, y):
    return -2 * x * (y - w @ x)


# gradient of the square loss
def logistic_grad(w, x, y):
    return (-y * x * np.exp(-y * (w @ x)))/(1 + np.exp(-y * (w @ x)))


# soft thresholding operator
def soft_thresholding(x, arg):
    w = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > arg:
            w[i] = x[i] - arg
        elif abs(x[i]) <= arg:
            w[i] = 0
        elif x[i] < -arg:
            w[i] = x[i] + arg

    return x


if __name__ == "__main__":
    main()

