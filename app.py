import logging
import telegram
import json
import random
import nltk
import pickle
import numpy as np
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from credentials import bot_token
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#Cargamos nuestros archivos
modelo = load_model('modelo.h5')
intentos = json.loads(open('intents.json',encoding="utf8").read())
palabras = pickle.load(open('palabras.pkl', 'rb'))
tags = pickle.load(open('tags.pkl', 'rb'))

#Configuración del loggin
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()
#TOKEN del bot
TOKEN = bot_token

def limpiarTexto(mensaje):
    msg_usuario = nltk.word_tokenize(mensaje)
    msg_usuario = [lemmatizer.lemmatize(palabra.lower()) for palabra in msg_usuario]
    return msg_usuario

def busqueda(mensaje, palabras):
    msg_usuario = limpiarTexto(mensaje)
    aux = [0]*len(palabras)

    for m in msg_usuario:
        for i,p in enumerate(palabras):
            if p == m:
                aux[i] = 1
    return(np.array(aux))

def predecirTag(mensaje, modelo):
    p = busqueda(mensaje, palabras)
    respuesta = modelo.predict(np.array([p]))[0]

    ERROR_THRESHOLD = 0.25
    resultados = [[i, r] for i,r in enumerate(respuesta) if r > ERROR_THRESHOLD]
    resultados.sort(key=lambda x: x[1], reverse=True)
    lista=[]
    
    for r in resultados:
        lista.append({"intent": tags[r[0]], "probability": str(r[1])})
    return lista
    

def obtenerRespuesta(intentos, intentos_json):
    tag = intentos[0]['intent']
    lista_intentos = intentos_json['intents']

    for i in lista_intentos:
        if i['tag'] == tag:
            resultado = random.choice(i['responses'])
            break
    return resultado

def predecirRespuesta(mensaje):
    ints = predecirTag(mensaje, modelo)
    res = obtenerRespuesta(ints, intentos)
    return res

def start(update, context):
    #print(update)
    logger.info(f"El usuario {update.effective_user['id']} inicio conversacion.")
    name = update.effective_user['first_name']
    update.message.reply_text(f"Hola {name}, yo soy PsicBot")

def responder(update, context):
    user_id = update.effective_user['id']
    logger.info(f"El usuario {user_id}, a enviado un mensaje de texto.")
    texto = update.message.text
    if texto != '':
        res = predecirRespuesta(texto)
        context.bot.send_message(
            chat_id=user_id,
            parse_mode="MarkdownV2",
            text=f"{res}"
        )

#Para obtener info del bot
if __name__=='__main__':
    my_bot = telegram.Bot(token=TOKEN)
    #print(my_bot.getMe())

#Enlace entre el updater con el bot
updater = Updater(my_bot.token, use_context=True)
#Despachador
dp= updater.dispatcher

#Manejadores
dp.add_handler(CommandHandler("start", start))
dp.add_handler(MessageHandler(Filters.text, responder))

updater.start_polling()
print("Bot activo")
updater.idle() #Para finalizar el bot con ctrl + c
