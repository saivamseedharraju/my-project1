
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

import pyttsx3
import speech_recognition as sr
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
print(voices[0].id)
engine.setProperty('voices',voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


#Creating GUI with tkinter
import tkinter
from tkinter import *
from ttkthemes import ThemedTk,THEMES




def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#ffffff", font=("Times New Roman", 14 ))
        
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        speak(res)
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
    





# base = Tk()
base= ThemedTk(themebg=True)
base.set_theme('black')
base.title("PULI")
base.geometry("400x500")
base.iconbitmap(r'logo.ico')
base.resizable(width=TRUE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="#0D0D0D", height="8", width="50", font="Arial")

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="circle")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#E83A59", activebackground="#6A1B4D",fg='#ffffff',
                    command= send )
def my_command():
    pass


def speakit():
    r= sr.Recognizer()
    r.pause_threshold=0.7
    r.energy_threshold=400
    with sr.Microphone() as source:
        
        try:
            audio=r.listen(source,timeout=5)
            message=str(r.recognize_google(audio,language="en-in"))
            EntryBox.focus()
            EntryBox.delete("0.0",END)
            EntryBox.insert("0.0",message)
        except Exception as e:
            return "none"

click_btn= PhotoImage(file='mic4.png')
img_label= Label(image=click_btn)
button= Button(base, image=click_btn,command=speakit,borderwidth=0)
button.pack(pady=30)

# text= Label(base, text= "")
# text.pack(pady=30)

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=200, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
button.place(x=128, y=401, height=90)
base.mainloop()
