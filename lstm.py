import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import History 

def runLstm(fName):
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)
        
    # fix random seed for reproducibility
    np.random.seed(42)
    
    # load the dataset
    dataframe= pd.read_csv(fName, usecols=[1], engine='python')
    dataset= dataframe.values
    dataset= dataset.astype('float64')
    
    # normalize the dataset
    scaler= MinMaxScaler(feature_range=(0,1))
    dataset= scaler.fit_transform(dataset)
    
    # split into train and test sets
    train_size= int(len(dataset))
    train= dataset[:,:]
    
    # reshape into X=t and Y=t+1
    max_epochs= 200
    look_back= 12
    trainX, trainY= create_dataset(train, look_back)
    
    # reshape input to be [samples, time steps, features]
    trainX= np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    if trainX.shape[0]%2!=0: 
        trainX= trainX[:-1]
        trainY= trainY[:-1]
        
    # create and fit the LSTM network
    min_batch_size= int(trainX.shape[0]*0.01)
    batch_size= 0
    cval= 1
    while batch_size < min_batch_size:
        if trainX.shape[0]%(min_batch_size+cval)==0: batch_size= min_batch_size+cval
        cval+= 1

    hist= History() # FOR EARLY STOPPING
    model= Sequential()
    model.add(LSTM(9, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(LSTM(6, stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=0, shuffle=False, callbacks=[hist])
    model.reset_states()
    
    print('BATCH_SIZE:', batch_size)
    hist_val= hist.history['loss'][0]
    for i in range(max_epochs):
        model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=0, shuffle=False, callbacks=[hist])
        model.reset_states()
        print('EPOCHS:', i+1, end='\r')
        '''
        if i > 3 and (np.abs(hist_val-hist.history['loss'][0])/hist_val < 0.001 or hist_val==0):
            print('TOTAL_EPOCHS:', i+1)
            break
        else: hist_val= hist.history['loss'][0]
        # print(hist_val)
        '''
    # make predictions
    trainPredict= model.predict(trainX, batch_size=batch_size)
    
    # invert predictions
    trainPredict= scaler.inverse_transform(trainPredict)
    trainY= scaler.inverse_transform([trainY])
    
    # calculate root mean squared error
    trainScore= math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    # print('Train Score: %.2f RMSE' % (trainScore))
    
    # shift train predictions for plotting
    trainPredictPlot= np.empty_like(dataset)
    trainPredictPlot[:,:]= np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back,:]= trainPredict

    return trainPredictPlot