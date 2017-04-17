import numpy as np
import re, tarfile, random
from functools import reduce
import keras
from keras.layers import Dense, Merge, Dropout, RepeatVector, Activation, recurrent
from keras.layers.recurrent import SimpleRNN
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.utils.data_utils import get_file
from keras.callbacks import History
from keras import backend as K

def sample(preds, temperature):
    # Helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

data = open('data/input.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'Unique characters:', chars
print 'The data has', data_size, 'characters, with', vocab_size, 'unique characters'
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
train_data=data[:int(0.8*data_size)]
val_data = data[int(0.8*data_size):]
seq_len = 25

train_data = [data[i] for i in range(len(data))]
train_data=train_data[:-30]

train_data_onehot = [list(to_categorical(char_to_ix[x],vocab_size)[0]) for x in train_data]
train_data_onehot = np.array(train_data_onehot)


training_batches = np.reshape(train_data_onehot, (int(train_data_onehot.shape[0]/seq_len), seq_len, vocab_size))

X = training_batches[:,:-1,:]
y = training_batches[:,1:,:]
train_len=int(0.8*training_batches.shape[0])
X_train = X[:train_len,:,:]
y_train = y[:train_len,:,:]
X_valid = X[train_len:,:,:]
y_valid = y[train_len:,:,:]

convert2String = lambda y: ''.join([ix_to_char[x[0]] for x in list(np.reshape(np.argmax(y, axis=2), (-1,1)))])

epochs=100
input_dim = vocab_size
hidden_dim = 100
output_dim = vocab_size
rnn_model = Sequential()
rnn_model.add(SimpleRNN(hidden_dim,
                        activation='tanh', return_sequences = True, input_shape = (None,vocab_size)))
rnn_model.add(Dense(output_dim))
rnn_model.add(Activation('softmax'))
rnn_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
rnn_model.summary()
print 'Training'

# For logging
# log_name = 'training_optimiser_'+ str(o) +'.txt'
# print 'Loss will be saved to', log_name
# csv_logger = keras.callbacks.CSVLogger(log_name, separator=',', append=False)

modelhistory = History()
rnn_model.fit(X_train,y_train, batch_size=50, nb_epoch=epochs, validation_data=(X_valid,y_valid))

# Function to get rnn layer output
get_rnn_layer_output = K.function([rnn_model.layers[0].input], [rnn_model.layers[0].output])

prime_len = 25
gen_len = 900
start_index = 0
d =0
rnn_activations = []
for T in [1.0]:
	d +=1
	generated = ''
	sentence = data[start_index: start_index + prime_len]
	generated += sentence
	print 'Generating with seed: "' + sentence + '"'

	for i in range(gen_len):
		x = np.zeros((1, prime_len, len(chars)))
		for t, char in enumerate(sentence):
			x[0, t, char_to_ix[char]] = 1.

		preds = rnn_model.predict(x, verbose=0)[0]
		layer_output = get_rnn_layer_output([x])[0]
		rnn_activations.append(layer_output[0][-1])
		next_index = sample(preds[-1], T)
		next_char = ix_to_char[next_index]

		generated += next_char
		sentence = sentence[1:] + next_char

	f= open('pred_feature' +'_'+ str(T)+ '_' + str(d) + '.txt','w')
	f.write(generated)
	f.close()
	rnn_activations = np.array(rnn_activations)
	print rnn_activations.shape
	np.savetxt('rnn_activations_pred',rnn_activations,delimiter =',')