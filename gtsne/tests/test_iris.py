def test_iris():
    from gtsne import gtsne
    from sklearn.datasets import load_iris

    iris = load_iris()

    X = iris.data
    # y = iris.target

    X_2d = gtsne(X,theta = 0.8, K=3, alpha=1e-2, beta=1e-6)

    assert X_2d.shape[0] == 150
    assert X_2d.shape[1] == 2

# test_iris()