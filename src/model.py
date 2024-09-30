from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape):
    model = Sequential()
    
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_model(model, X_train, y_train, epochs=25, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
def save_model(model, model_path):
    model.save(model_path)

def load_model(model_path):
    from keras.models import load_model
    return load_model(model_path)
