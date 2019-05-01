import keras
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import warnings
warnings.simplefilter('ignore')

#Set of AI models using Keras API

#Artificial Neural Network Model
class ANN():
    def __init__(
            self,
            network_sizes,
            activation = None,
            dropout = 0,
            optimizer = 'adam',
            loss = 'mean_squared_error',
            metrics = None,
            activation_out = None,
            n_out = 1,
            softmax = True,
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
        self.model.add(keras.layers.Dense(n_out,activation = self.activation_out))
        
        #If logits regression create softmax
        if n_out > 1 and self.softmax:
            self.model.add(keras.layers.Softmax())
        
        #Compile model
        self.model.compile(loss = self.loss, optimizer=self.optimizer, metrics = self.metrics)
        

        
    def fit_model(
            self,
            x_train,
            y_train,
            batch_size,
            epochs,
            validation_split = None,
            x_test = None,
            y_test = None,
            verbose = 1
            ):
        
        self.__dict__.update(locals())
                
        #Fit model
        if self.validation_split != None:
            self.history = self.model.fit(
              x = self.x_train,
              y = self.y_train,
              batch_size=self.batch_size,
              epochs=self.epochs,
              verbose=self.verbose,
              validation_split = self.validation_split
             )
            
        else:
            self.history = self.model.fit(
              x = self.x_train,
              y = self.y_train,
              batch_size=self.batch_size,
              epochs=self.epochs,
              verbose=self.verbose,
              validation_data = (self.x_test,self.y_test)
             ) 
        
    def predict(self,x):
        return self.model.predict(x)
    
    def eval(self,x,y):
        return  self.model.evaluate(x, y, verbose=self.verbose)
    
    def plot_loss(self):        
        frame = pd.DataFrame(data = np.array([self.model.history.history['val_loss'],
          self.model.history.history['loss']]).transpose(),columns=['Test','Train'])

        sns.lineplot(data = frame, palette="tab10",linewidth=2.5)
        
        plt.show()
        
    def plot_acc(self):
        try:            
            frame = pd.DataFrame(data = np.array([self.model.history.history['val_acc'],
               self.model.history.history['acc']]).transpose(),columns=['Test','Train'])

            sns.lineplot(data = frame, palette="tab10",linewidth=2.5)
            
            plt.show()
        except: 
            print('No accuracy on model')
            
    def plot_scatter(self,x,y):
        
        pred = self.model.predict(x)
        
        ax_x = pred.reshape((-1))
        
        ax_y = y.reshape((-1))
        
        sns.jointplot(x = ax_x, y = ax_y,
              kind='reg', color='b').set_axis_labels('Prediction','Actual')
        
        join_arr = np.concatenate((ax_x,ax_y))
        mini = np.min(join_arr)
        maxi = np.max(join_arr)             
        plt.plot([mini,maxi],[mini,maxi],'g')              
                      
        plt.show()
        
    def plot_scatter_var(self,x,y):
        
        pred = self.model.predict(x)
        
        ax_x = (pred - x[:,-1:]).reshape(-1)
        ax_y = (y-x[:,-1:]).reshape(-1)
        
        sns.jointplot(x = ax_x, y=ax_y,
              kind='reg', color='b').set_axis_labels('Actual','Prediction')
        
        join_arr = np.concatenate((ax_x,ax_y))
        mini = np.min(join_arr)
        maxi = np.max(join_arr)             
        plt.plot([mini,maxi],[mini,maxi],'g')
        
        plt.show()
        
    def plot_logits(self,x,y):
        
        pred = self.model.predict(x)
        
        df = pd.DataFrame({'actual':np.argmax(y, axis = 1),
                   'prediction':np.argmax(pred, axis = 1)})
        
        c = df.groupby('actual')['prediction'].value_counts().unstack().fillna(0)
        
        sns.heatmap(c,robust = True, annot = True)
        
        plt.show()
        
        
        
        
            
        
        
#Convolutional Neural Network        
class CNN():
    def __init__(
            self,
            network_sizes,
            filters,
            kernels,
            pool_size = (1,1),
            activation = None,
            activation_conv = None,
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
                f, kernel_size= k ,activation = self.activation_conv,))
     
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
            x_train,
            y_train,
            batch_size,
            epochs,
            validation_split = None,
            x_test = None,
            y_test = None,
            verbose = 1
            ):
        
        self.__dict__.update(locals())
        
        #Fit model
        if self.validation_split != None:
            self.history = self.model.fit(
              x_train = self.x_train,
              y_train = self.y_train,
              batch_size=self.batch_size,
              epochs=self.epochs,
              verbose=self.verbose,
              validation_split = self.validation_split
             )
            
        else:
            self.history = self.model.fit(
              x = self.x_train,
              y = self.y_train,
              batch_size=self.batch_size,
              epochs=self.epochs,
              verbose=self.verbose,
              validation_data = (self.x_test,self.y_test)
             )  
        
        
    def predict(self,x):
        return self.model.predict(x)
    
    def eval(self,x,y):
        return  self.model.evaluate(x, y, verbose=self.verbose)

    def plot_loss(self):        
        frame = pd.DataFrame(data = np.array([self.model.history.history['val_loss'],
          self.model.history.history['loss']]).transpose(),columns=['Test','Train'])

        sns.lineplot(data = frame, palette="tab10",linewidth=2.5)
        
        plt.show()
        
    def plot_acc(self):
        try:            
            frame = pd.DataFrame(data = np.array([self.model.history.history['val_acc'],
               self.model.history.history['acc']]).transpose(),columns=['Test','Train'])

            sns.lineplot(data = frame, palette="tab10",linewidth=2.5)
            
            plt.show()
        except: 
            print('No accuracy on model')
    
    def plot_scatter(self,x,y):
        
        pred = self.model.predict(x)
        
        ax_x = pred.reshape((-1))
        
        ax_y = y.reshape((-1))
        
        sns.jointplot(x = ax_x, y = ax_y,
              kind='reg', color='b').set_axis_labels('Prediction','Actual')
        
        join_arr = np.concatenate((ax_x,ax_y))
        mini = np.min(join_arr)
        maxi = np.max(join_arr)             
        plt.plot([mini,maxi],[mini,maxi],'g')              
                      
        plt.show()    
    
    def plot_logits(self,x,y):
        
        pred = self.model.predict(x)
        
        df = pd.DataFrame({'actual':np.argmax(y, axis = 1),
                   'prediction':np.argmax(pred, axis = 1)})
        
        c = df.groupby('actual')['prediction'].value_counts().unstack().fillna(0)
        
        sns.heatmap(c,robust = True, annot = True)
        
        plt.show()
        
        
    
    
    
    
    
    
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
            x_train,
            y_train,
            batch_size,
            epochs,
            validation_split = None,
            x_test = None,
            y_test = None,
            verbose = 1
            ):
        
        self.__dict__.update(locals())

        #Fit model
        if self.validation_split != None:
            self.history = self.model.fit(
              x = self.x_train,
              y = self.y_train,
              batch_size=self.batch_size,
              epochs=self.epochs,
              verbose=self.verbose,
              validation_split = self.validation_split
             )
            
        else:
            self.history = self.model.fit(
              x = self.x_train,
              y = self.y_train,
              batch_size=self.batch_size,
              epochs=self.epochs,
              verbose=self.verbose,
              validation_data = (self.x_test,self.y_test)
             )       
        
    def predict(self,x):
        return self.model.predict(x)
    
    def eval(self,x,y):
        return  self.model.evaluate(x, y, verbose=self.verbose)
    
    def plot_loss(self):        
        frame = pd.DataFrame(data = np.array([self.model.history.history['val_loss'],
          self.model.history.history['loss']]).transpose(),columns=['Test','Train'])

        sns.lineplot(data = frame, palette="tab10",linewidth=2.5)
        
        plt.show()
        
    def plot_acc(self):
        try:            
            frame = pd.DataFrame(data = np.array([self.model.history.history['val_acc'],
               self.model.history.history['acc']]).transpose(),columns=['Test','Train'])

            sns.lineplot(data = frame, palette="tab10",linewidth=2.5)
            
            plt.show()
        except: 
            print('No accuracy on model')    
    
    def plot_scatter(self,x,y):
        
        pred = self.model.predict(x)
        
        ax_x = pred.reshape((-1))
        
        ax_y = y.reshape((-1))
        
        sns.jointplot(x = ax_x, y = ax_y,
              kind='reg', color='b').set_axis_labels('Prediction','Actual')
        
        join_arr = np.concatenate((ax_x,ax_y))
        mini = np.min(join_arr)
        maxi = np.max(join_arr)             
        plt.plot([mini,maxi],[mini,maxi],'g')              
                      
        plt.show()
        
    def plot_scatter_var(self,x,y):
        
        pred = self.model.predict(x)
        
        ax_x = (pred - x[:,-1:].reshape(-1,1)).reshape(-1)
        ax_y = (y-x[:,-1:].reshape(-1,1)).reshape(-1)
        
        sns.jointplot(x = ax_x, y=ax_y,
              kind='reg', color='b').set_axis_labels('Actual','Prediction')
        
        join_arr = np.concatenate((ax_x,ax_y))
        mini = np.min(join_arr)
        maxi = np.max(join_arr)             
        plt.plot([mini,maxi],[mini,maxi],'g')
        
        plt.show()
        
    def plot_logits(self,x,y):
        
        pred = self.model.predict(x)
        
        df = pd.DataFrame({'actual':np.argmax(y, axis = 1),
                   'prediction':np.argmax(pred, axis = 1)})
        
        c = df.groupby('actual')['prediction'].value_counts().unstack().fillna(0)
        
        sns.heatmap(c,robust = True, annot = True)
        
        plt.show()    
    
    
    
    
    
    
    
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
                    activation = self.activation,return_sequences = True
                                                 #,stateful=True,batch_input_shape=(1)
                                                ))
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
            x_train,
            y_train,
            batch_size,
            epochs,
            validation_split = None,
            x_test = None,
            y_test = None,
            verbose = 1,
            ):
        
        self.__dict__.update(locals())
        
        if self.crop[1] == 0:
            self.y_train = self.y_train[:,self.crop[0]:]
            self.y_test = self.y_test[:,self.crop[0]:]
        else:
            self.y_train = self.y_train[:,self.crop[0]:-self.crop[1]]
            self.y_test = self.y_test[:,self.crop[0]:-self.crop[1]]
            
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.verbose = verbose
        
        #Fit model
        if self.validation_split != None:
            self.history = self.model.fit(
              x = self.x_train,
              y = self.y_train,
              batch_size = self.batch_size,
              epochs = self.epochs,
              verbose = self.verbose,
              validation_split = self.validation_split
             )
        else:
            self.history = self.model.fit(
              x = self.x_train,
              y = self.y_train,
              batch_size = self.batch_size,
              epochs = self.epochs,
              verbose = self.verbose,
              validation_data = (self.x_test,self.y_test)
             )
        
    def predict(self,x):
        return self.model.predict(x)
    
    def eval(self,x,y):
        return  self.model.evaluate(
            x, y[:,self.crop[0]:-self.crop[1]], verbose=self.verbose)
    
    def plot_loss(self):        
        frame = pd.DataFrame(data = np.array([self.model.history.history['val_loss'],
          self.model.history.history['loss']]).transpose(),columns=['Test','Train'])

        sns.lineplot(data = frame, palette="tab10",linewidth=2.5)
        
        plt.show()
        
    def plot_acc(self):
        try:            
            frame = pd.DataFrame(data = np.array([self.model.history.history['val_acc'],
               self.model.history.history['acc']]).transpose(),columns=['Test','Train'])

            sns.lineplot(data = frame, palette="tab10",linewidth=2.5)
            
            plt.show()
        except: 
            print('No accuracy on model')
    
    def plot_scatter(self,x,y):
        
        pred = self.model.predict(x)
        
        ax_x = pred.reshape((-1))
        
        if self.crop[1] == 0:
            ax_y = y[:,self.crop[0]:].reshape(-1)
            
        else:
            ax_y = y[:,self.crop[0]:-crop[1]].reshape(-1) 
                          
                
        sns.jointplot(x = ax_x, y = ax_y,
              kind='reg', color='b').set_axis_labels('Prediction','Actual')
        
        join_arr = np.concatenate((ax_x,ax_y))
        mini = np.min(join_arr)
        maxi = np.max(join_arr)             
        plt.plot([mini,maxi],[mini,maxi],'g')              
                      
        plt.show()
        
    def plot_scatter_var(self,x,y):
        
        pred = self.model.predict(x)
        
        
        if self.crop[1] == 0:
            ax_x = (pred - x[:,self.crop[0]:]).reshape(-1)
            ax_y = (y-x)[:,self.crop[0]:].reshape(-1)
            
        else:
            ax_x = (pred - x[:,self.crop[0]:-self.crop[1]]).reshape(-1)
            ax_y = (y-x)[:,self.crop[0]:-self.crop[1]].reshape(-1)
        
        sns.jointplot(x = ax_x, y=ax_y,
              kind='reg', color='b').set_axis_labels('Actual','Prediction')
        
        join_arr = np.concatenate((ax_x,ax_y))
        mini = np.min(join_arr)
        maxi = np.max(join_arr)             
        plt.plot([mini,maxi],[mini,maxi],'g')
        
        plt.show()
        

#Function to plot in a grid
def plot_grid(ims, shape, cmap = None):
    
    selection = np.random.choice( range(len(ims) ), shape ,replace=False)

    plt.figure(1)
    
    n = 1
    
    for i in range( shape[0] ):
        
        for j in range( shape[1] ):
            
            plt.subplot(shape[0], shape[1], n)
            
            plt.axis('off')
            
            plt.imshow(ims[ selection[i][j] ], interpolation='nearest', cmap = cmap)
            
            n += 1
            
    
    plt.show()        
        
        
class GAN():
    def __init__(
        self, 
        generator, 
        discriminator,
        optimizer
        ):
        #Pass elements from init
        self.__dict__.update(locals())
        #Make discriminator not trainable
        self.discriminator.trainable = False
        #Create gan joining discriminator and generator
        self.gan = keras.models.Sequential(layers = [generator,discriminator])
        #Compile model
        self.gan.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        #Current iteration on model
        self.round = 0
        #Loss on generator
        self.loss_gen = []
        #Loss on discriminator
        self.loss_disc = []
        
    def train_gan(self,
                  noise_function,
                  actual,
                  smoothing = 1,
                  verbose = 1,
                  actual_reshape = None,
                  epochs_discr = 1,
                  epochs_gen = 1,
                  size_disc = None,
                  batch_size_disc = None,
                  size_gen = None,
                  batch_size_gen = None,
                 ):
        
        self.__dict__.update(locals())
        
        ###Show round
        if self.verbose == 1: print('Round: ', self.round)
            
        ###Select from actual (batch size)
        if self.size_disc != None:
            ind = np.random.choice(
                range(len(self.actual)),self.size_disc, replace = False)
            actual = self.actual[ind]
        else:
            actual = self.actual

            
        ###Create noise    
        self.noise = self.noise_function(len(actual))
        ###Generate from noise
        self.generated = self.generator.predict(self.noise)
        ###Reshape data if necessary
        if self.actual_reshape != None: 
            self.generated = self.generated.reshape(self.actual_reshape)
        
        ###Train discriminator
        
        ##Create labels
        labels = np.concatenate(
            [ np.ones(len(actual))*self.smoothing ,
             np.zeros(len(self.noise)) ]).reshape(-1,1)
        
        ##Create X joining actual and generated
        x = np.concatenate([actual, self.generated])
        
        ###Train discriminator
        self.discriminator.trainable = True
        if self.verbose == 1: print('Training Discriminator:')
        self.discriminator.fit(
            x = x,
            y = labels,
            verbose = self.verbose,
            epochs = self.epochs_discr,
            batch_size = self.batch_size_disc
            )
        #Append discriminator loss
        self.loss_disc.append(self.discriminator.history.history['loss'])
        
        ###Select from actual (batch size)
        if self.size_gen != None:
            ind = np.random.choice(
                range(len(self.actual)),self.size_gen, replace = False)
            actual = self.actual[ind]
        else:
            actual = self.actual
            
        ###Create noise    
        self.noise = self.noise_function(len(actual))
        
        ###Train generator
        if self.verbose == 1: print('Training Generator:')
        self.discriminator.trainable = False
        self.generator.trainable = True
        
        ###Fit GAN model
        self.gan.fit(self.noise, np.ones((len(self.noise),1)),
                     epochs = self.epochs_gen, verbose = self.verbose,
                    batch_size = self.batch_size_gen)
        #Append generator loss
        self.loss_disc.append(self.gan.history.history['loss'])
        
        self.round += 1
        
    def predict_gan(self,noise):
        
        return self.generator.predict(noise)
    
    def train_batch(self,
                  noise_function,
                  actual,
                  n_do = 1,
                  smoothing = 1,
                  verbose = 1,
                  actual_reshape = None,
                  epochs_discr = 1,
                  epochs_gen = 1,
                  size_disc = None,
                  batch_size_disc = None,
                  size_gen = None,
                  batch_size_gen = None,
                  n_show = None,
                  show_reshape = None,
                  cmap = None,
                  grid_show = (2,2)
                    
                   ):
        
        self.__dict__.update(locals())        
        
        for i in range(self.n_do):
        
            self.train_gan(
                  noise_function = self.noise_function,
                  actual = self.actual,
                  smoothing = self.smoothing,
                  verbose = self.verbose,
                  actual_reshape = self.actual_reshape,
                  epochs_discr = self.epochs_discr,
                  epochs_gen = self.epochs_gen,
                  size_disc = self.size_disc,
                  batch_size_disc = self.batch_size_disc,
                  size_gen = self.size_gen,
                  batch_size_gen = self.batch_size_gen,
            )
            
            if n_show != None:
                
                if i % n_show == 0:
                
                    print('Round: ',self.round)

                    noise = self.noise_function(np.prod(grid_show))

                    generated = self.predict_gan(noise)

                    plot_grid(generated.reshape(show_reshape),
                                   self.grid_show,cmap = self.cmap)
                    

        

        
        


        
        
        
        
        