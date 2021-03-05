import tensorflow.keras as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
from tensorflow.keras.layers import Dense,LeakyReLU,BatchNormalization,Input


class AutoEncoder():
    def __init__(self, f_dims, new_dims):
        self.f_dims=f_dims
        self.new_dims=f_dims[1] if new_dims is None else new_dims


    def dense_layer(self, x, size=1):
        x = Dense(self.f_dims[1]*size)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x


    def encoder(self,x):
        e1 = self.dense_layer(x, 2)
        e2 = self.dense_layer(x)
        return e2


    def decoder(self,x):
        d1 = self.dense_layer(x)
        d2 = self.dense_layer(x, 2)
        return d2


    def build(self):

        x_in = Input((self.f_dims[1],))

        x = self.encoder(x_in)
        x = Dense(self.new_dims)(x)
        x = self.decoder(x)
        x_out = Dense(self.f_dims[1], activation='linear')(x)
        model = Model(x_in,x_out)

        return model
