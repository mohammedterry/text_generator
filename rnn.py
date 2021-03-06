import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop

def clean(text):
    return ' '.join(''.join([ch.lower() if 96 < ord(ch) < 123 else ' ' for ch in text.lower()]).split())

def window_transform_text(text, window_size, step_size=5):
    inputs = [text[i:i+window_size] for i in range(0,len(text)-window_size,step_size)]
    outputs = [text[i+window_size] for i in range(0,len(text)-window_size,step_size)]
    return inputs,outputs

def encode_io_pairs(text,window_size,step_size=5):
    n = 28
    inputs, outputs = window_transform_text(text,window_size,step_size)
    X = np.zeros((len(inputs), window_size, n), dtype=np.bool)
    y = np.zeros((len(inputs), n), dtype=np.bool)
    for i, sentence in enumerate(inputs):
        for t, char in enumerate(sentence):
            X[i, t, chars_to_indices(char)] = 1
        y[i, chars_to_indices(outputs[i])] = 1     
    return X,y

def predict_next_chars(model,input_chars,num_to_predict):     
    n = 28
    predicted_chars = ''
    for i in range(num_to_predict):
        x_test = np.zeros((1, window_size, n))
        for t, char in enumerate(input_chars):
            x_test[0, t, chars_to_indices(char)] = 1.
        test_predict = model.predict(x_test,verbose = 0)[0]
        r = np.argmax(test_predict)                           
        d = indices_to_chars(r) 
        predicted_chars+=d
        input_chars+=d
        input_chars = input_chars[1:]
    return predicted_chars

def indices_to_chars(i):
    if i == 0:
        return ' '
    return chr(i + 96)

def chars_to_indices(ch):
    if 96 < ord(ch.lower()) < 123:
        return ord(ch.lower()) - 96
    return 0


# CLEAN INPUT TEXT
n = 28
window_size = 15
text = clean(open('twitter.txt').read())[:100000]
print ("\n{} ...\n\nthis corpus has: \n\t{} characters in total\n\t{} unique characters\n\t\t{}".format(text[:100],len(text),len(set(text)),sorted(list(set(text)))))
X,y = encode_io_pairs(text,window_size)
# CREATE LSTM
model = Sequential()
model.add(LSTM(200,input_shape=(window_size, n )))
model.add(Dense(n))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=.001,rho=.9,epsilon=1e-08,decay=.0))
model.load_weights('best_RNN_large_textdata_weights.hdf5')
# TRAIN LSTM / LEARN TEXT
model.fit(X, y, batch_size=500, epochs=30, verbose=1)
model.save_weights('best_RNN_large_textdata_weights.hdf5')
# TEST LSTM / GENERATE TEXT
predict_input = predict_next_chars(model,input()[-window_size:],num_to_predict = 100)
print('-------------------\npredicted chars = \n {} "\n'.format(predict_input))
