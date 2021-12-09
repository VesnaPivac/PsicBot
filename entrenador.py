import nltk
import json
import pickle
import numpy as np
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random

#definimos nuestas variables
palabras=[]
tags=[]
documentos=[]
ignorar=['?','!']
archivo=open('intents.json',encoding="utf8").read()
intentos = json.loads(archivo)

#Primero, importamos nuestros 'intents'
for intento in intentos['intents']:
    for patrones in intento['patterns']:
        #tokenizamos cada palabra de nuestras preguntas
        p = nltk.word_tokenize(patrones)
        #Agregamos p a nuestras palabras
        palabras.extend(p)
        #añadimos a documentos la dupla de (palabra tokenizada, nombre de la clase que pertenece)
        documentos.append((p, intento['tag']))

        #agregamos los tags a tags para categorizarlas
        if intento['tag'] not in tags:
            tags.append(intento['tag'])

#limpiamos las palabras. En este caso solo son los caracteres '?' y '!' y ordenamos
palabras = [lemmatizer.lemmatize(p.lower()) for p in palabras if p not in ignorar]
palabras = sorted(list(set(palabras)))
tags = sorted(list(set(tags)))

#Volvemos nuestras variables 'dummys'
pickle.dump(palabras, open('palabras.pkl', 'wb'))
pickle.dump(tags, open('tags.pkl', 'wb'))

#Ahora crearemos nuestros datos de entrenamiento
datos_entrenamiento = []
lista_aux = [0] * len(tags) 
for doc in documentos:
    aux=[]
    preguntas= doc[0]
    #convertimos a minusculas nuestras palabras
    preguntas = [lemmatizer.lemmatize(p.lower()) for p in preguntas]

    #agregamosnlas palabras claves que se encuentran en nuestras preguntas
    for p in palabras:
        aux.append(1) if p in preguntas else aux.append(0)

    salida = list(lista_aux)
    salida[tags.index(doc[1])] = 1
    datos_entrenamiento.append([aux, salida])

#entrenamos
random.shuffle(datos_entrenamiento)
datos_entrenamiento = np.array(datos_entrenamiento)
train_x = list(datos_entrenamiento[:,0])
train_y = list(datos_entrenamiento[:,1])
print("training data ceated")

#Creamos nuestra red neuronal de 3 capas
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0],), activation='softmax'))
print(model.summary())

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Guardamos nuestro modelo y listo
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('modelo.h5', hist)
print("model created")
