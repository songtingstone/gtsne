def test_seed():
    from gtsne import gtsne
    from sklearn.datasets import load_iris
    import numpy as np

    iris = load_iris()

    X = iris.data
    # y = iris.target

    t1 = gtsne(X, random_state=np.random.RandomState(0), copy_data=True)
    # t2 = gtsne(X, random_state=np.random.RandomState(0), copy_data=True)

    assert t1.shape[0] == 150
    assert t1.shape[1] == 2
    # assert np.all(t1 == t2)
