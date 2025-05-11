# code based on https://github.com/harshilpatel312/KITTI-distance-estimation
from sklearn.preprocessing import StandardScaler


def infer_dist(loaded_model, X_test=None, y_test=None):

    # standardized data
    scalar = StandardScaler()
    X_test = scalar.fit_transform(X_test)
    y_test = scalar.fit_transform(y_test)  # necessary


    y_pred = loaded_model.predict(X_test)

    # scale up predictions to original values
    y_pred = scalar.inverse_transform(y_pred)
    # y_test = scalar.inverse_transform(y_test)
    return y_pred[0][0]  # the format is [[]]

if __name__ == '__main__':
    infer_dist()
