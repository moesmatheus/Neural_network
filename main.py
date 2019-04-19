import keras

#Set of AI models using Keras API

#Artificial Neural Network Model
class ANN():
    def __init__(
            self,
            network_sizes,
            activation=None,
            dropout=0,
            optimizer = 'adam',
            loss = 'mean_squared_error',
            metrics = None,
            n_out = 1
           ):
        
        #Pass elements from init
        self.__dict__.update(locals())
        
        #Create model
        self.model = keras.models.Sequential()
        
        #Create Hidden Layers
        for i in self.network_sizes:
            self.model.add(keras.layers.Dropout(self.dropout))
            self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.Dense(i,activation = self.activation))
        
        #Create output layer
        self.model.add(keras.layers.Dense(n_out))
        
        #If logits regression create softmax
        if n_out > 1:
            self.model.add(keras.layers.Softmax())
        
        #Compile model
        self.model.compile(loss = self.loss, optimizer=self.optimizer, metrics = self.metrics)
        

        
    def fit_model(
            self,
            x,
            y,
            batch_size,
            epochs,
            validation_split,
            verbose = 1
            ):
        
        self.x_train = x
        self.y_train = y
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.verbose = verbose
        
        #Fit model
        self.model.fit(
          x = self.x_train,
          y = self.y_train,
          batch_size=self.batch_size,
          epochs=self.epochs,
          verbose=self.verbose,
          validation_split = self.validation_split
         )
        
    def predict(self,x):
        return self.model.predict(x)
    
    def eval(self,x,y):
        return  self.model.evaluate(x, y, verbose=self.verbose)
        
        
#Convolutional Neural Network        
class CNN():
    def __init__(
            self,
            network_sizes,
            filters,
            kernels,
            pool_size = (1,1),
            activation=None,
            dropout=0,
            optimizer = 'adam',
            loss = 'mean_squared_error',
            metrics = None,
            n_out = 1
           ):
        
        #Pass elements from init
        self.__dict__.update(locals())
        
        #Create model
        self.model = keras.models.Sequential()
        
        #Create Convolution
        for f,k in zip(self.filters,self.kernels):
            self.model.add(keras.layers.Conv2D(
                f, kernel_size= k ,activation = self.activation,))
     
        #Create MaxPooling
        self.model.add(keras.layers.MaxPooling2D(pool_size=self.pool_size))
        self.model.add(keras.layers.Dropout(self.dropout))
        self.model.add(keras.layers.Flatten())
        
        #Create Hidden Layers
        for i in self.network_sizes:
            self.model.add(keras.layers.Dropout(self.dropout))
            self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.Dense(i,activation = self.activation))
        
        #Create output layer
        self.model.add(keras.layers.Dense(n_out))
        
        #If logits regression create softmax
        if n_out > 1:
            self.model.add(keras.layers.Softmax())
        
        #Compile model
        self.model.compile(loss = self.loss, optimizer=self.optimizer, metrics = self.metrics)
        
        
    def fit_model(
            self,
            x,
            y,
            batch_size,
            epochs,
            validation_split,
            verbose = 1
            ):
        
        self.x_train = x
        self.y_train = y
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.verbose = verbose
        
        #Fit model
        self.model.fit(
          x = self.x_train,
          y = self.y_train,
          batch_size=self.batch_size,
          epochs=self.epochs,
          verbose=self.verbose,
          validation_split = self.validation_split
         )
        
    def predict(self,x):
        return self.model.predict(x)
    
    def eval(self,x,y):
        return  self.model.evaluate(x, y, verbose=self.verbose)

#Recurrent Neural Network
class RNN():
    def __init__(
            self,
            network_sizes,
            lstm_cells,
            rnn_type = 'lstm',
            activation=None,
            dropout=0,
            optimizer = 'adam',
            loss = 'mean_squared_error',
            metrics = None,
            n_out = 1
           ):
        
        #Pass elements from init
        self.__dict__.update(locals())
        
        #Create model
        self.model = keras.models.Sequential()
        
        #Create LSTM Layer
        for i in self.lstm_cells:
            if self.rnn_type == 'lstm':
                self.model.add(keras.layers.LSTM(i,dropout = self.dropout, 
                    activation = self.activation))
            else:
                self.model.add(keras.layers.GRU(i,dropout = self.dropout, 
                    activation = self.activation))
        
        #Create Hidden Layers
        for i in self.network_sizes:
            self.model.add(keras.layers.Dropout(self.dropout))
            self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.Dense(i,activation = self.activation))
        
        #Create output layer
        self.model.add(keras.layers.Dense(n_out))
        
        #If logits regression create softmax
        if n_out > 1:
            self.model.add(keras.layers.Softmax())
        
        #Compile model
        self.model.compile(loss = self.loss, optimizer=self.optimizer, metrics = self.metrics)
        
        
    def fit_model(
            self,
            x,
            y,
            batch_size,
            epochs,
            validation_split,
            verbose = 1
            ):
        
        self.x_train = x
        self.y_train = y
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.verbose = verbose
        
        #Fit model
        self.model.fit(
          x = self.x_train,
          y = self.y_train,
          batch_size=self.batch_size,
          epochs=self.epochs,
          verbose=self.verbose,
          validation_split = self.validation_split
         )
        
    def predict(self,x):
        return self.model.predict(x)
    
    def eval(self,x,y):
        return  self.model.evaluate(x, y, verbose=self.verbose)
    
    
#Recurrent Neural Network
class RNN_multistep():
    def __init__(
            self,
            network_sizes,
            lstm_cells,
            rnn_type = 'lstm',
            activation=None,
            dropout=0,
            crop = (0,0),
            optimizer = 'adam',
            loss = 'mean_squared_error',
            metrics = None,
            n_out = 1
           ):
        
        #Pass elements from init
        self.__dict__.update(locals())
        
        #Create model
        self.model = keras.models.Sequential()
        
        #Create LSTM Layer
        for i in self.lstm_cells:
            if self.rnn_type == 'lstm':
                self.model.add(keras.layers.LSTM(i,dropout = self.dropout, 
                    activation = self.activation,return_sequences = True))
            else:
                self.model.add(keras.layers.GRU(i,dropout = self.dropout, 
                    activation = self.activation,return_sequences = True))
        
        #Create Hidden Layers
        for i in self.network_sizes:
            self.model.add(keras.layers.Dropout(self.dropout))
            self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.Dense(i,activation = self.activation))
        
        #Create output layer
        self.model.add(keras.layers.Dense(n_out))
        
        #If logits regression create softmax
        if n_out > 1:
            self.model.add(keras.layers.Softmax())
        
        #Crop array
        self.model.add(keras.layers.Cropping1D(cropping=self.crop))
        
        #Compile model
        self.model.compile(loss = self.loss, optimizer=self.optimizer, metrics = self.metrics)
        
        
    def fit_model(
            self,
            x,
            y,
            batch_size,
            epochs,
            validation_split,
            verbose = 1
            ):
        
        self.x_train = x
        self.y_train = y
        if self.crop[1] == 0:
            self.y_train = self.y_train[:,self.crop[0]:]
        else:
            self.y_train = self.y_train[:,self.crop[0]:-self.crop[1]]
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.verbose = verbose
        
        #Fit model
        self.model.fit(
          x = self.x_train,
          y = self.y_train,
          batch_size=self.batch_size,
          epochs=self.epochs,
          verbose=self.verbose,
          validation_split = self.validation_split
         )
        
    def predict(self,x):
        return self.model.predict(x)
    
    def eval(self,x,y):
        return  self.model.evaluate(
            x, y[:,self.crop[0]:-self.crop[1]], verbose=self.verbose)