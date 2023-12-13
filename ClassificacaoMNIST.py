import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.optimizers import SGD
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random as rd

#carregar o dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#mostrar o numero de amostras, suas dimensões e rotulos que há nos sub-
# datasets de treino e teste
print("Total training sample and image size:", x_train.shape)
print("Total training number of labels:", y_train.shape)
print("Total training sample and image size:", x_test.shape)
print("Total training number of labels:", y_test.shape)

#Gerar imagens aleatórias do dataset para verificação de funcionamento
image_index = rd.randrange(59999) # escolher uma amostra aleatória: 
                                  #0 a 59999 totalizam 60000 amostras
print(y_train[image_index]) # printar em tela o rótulo dessa amostra
plt.imshow(x_train[image_index], cmap='Greys') #mostrar a imagem

#preparação dos dados
#reshape: adicionar dimensão de channel à imagem, necessária para 
# o processamento da imagem pela rede neural
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#converter para float: padronizar os dados todos para float; diminui gasto de memória
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#normalizar: necessário em modelos de rede neural; faz com que o valor 
# de cada pixel esteja entre 0 e 1
x_train = x_train/255
x_test = x_test/255

#one-hot encoding (usar somente com sgd)
#y_train = keras.utils.to_categorical(y_train)
#y_test = keras.utils.to_categorical(y_test)


#criar o modelo

# Sequential(): diz que o modelo é de camadas sequênciais
model = Sequential()
#adicionar camadas
model.add(Conv2D(64, kernel_size=(5,5), activation='relu', input_shape=(28,28,1))) 
#64 neuronios, filtro 5x5, ReLU de ativação, dimensões da entrada
model.add(MaxPooling2D(pool_size=(2, 2))) #filtro de pooling 2x2
model.add(Flatten()) #2D->1D
model.add(Dense(128, activation='relu')) #128 neuronios, ReLU de ativação
model.add(Dropout(0.2)) #dropout de 20% dos neuronios da camada anterior
model.add(Dense(10,activation='softmax')) #10 neuronios(1 p/ cada numero),
                                          # softmax de ativação
model.summary() #sumario das camadas do modelo


#Otimização e Compilação

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#opt = SGD(learning_rate=0.01, momentum=0.9)
#model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10) #10 passagens de treinamento do modelo
print("")
model.evaluate(x_test, y_test) #precisao do modelo

#predição
image_index = rd.randrange(9999) #0 a 9999 imagens de teste = 10000
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys') #formatação da imagem
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1)) #previsão
print('Prediction:', pred.argmax()) #argmax puxa resultado do softmax
print('Label:', y_test[image_index]) #print pra comparação da label real