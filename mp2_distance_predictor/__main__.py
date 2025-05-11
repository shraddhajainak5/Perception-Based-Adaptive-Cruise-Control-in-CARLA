from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import pandas as pd

def main():
    # ----------- Import data and scale ----------- #
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
    y_train = df_train[['zloc']].values

    X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
    y_test = df_test[['zloc']].values

    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    y_scalar = StandardScaler()
    y_train = y_scalar.fit_transform(y_train)
    y_test = y_scalar.transform(y_test)

    model = Sequential()
    model.add(Dense(64, input_dim=4, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='he_normal', activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10,
                                       verbose=1, min_delta=1e-4, mode='min')
    modelname = "distance_model"
    tensorboard = TensorBoard(log_dir=f"training_logs/{modelname}")

    history = model.fit(X_train, y_train,
                        validation_split=0.1,
                        epochs=500,
                        batch_size=512,
                        callbacks=[earlyStopping, reduce_lr_loss, tensorboard],
                        verbose=1)

    model_json = model.to_json()
    with open(f"/home1/jainak/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{modelname}.json", "w+") as json_file:
        json_file.write(model_json)

    model.save_weights(f"/home1/jainak/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{modelname}.h5")
    print("Saved model to disk")

if __name__ == '__main__':
    main()
