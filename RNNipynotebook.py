
# coding: utf-8

# In[83]:

import numpy as np
import os
from random import shuffle
import re
import tensorflow as tf;
import numpy as np;


from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
from gensim.models import Word2Vec
from operator import add

import urllib.request
import zipfile
import progressbar
import lxml.etree


# Download the dataset if it's not already there: this may take a minute as it is 75MB
if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")


# In[33]:


train_label = [];
validation_label = [];
test_label = [];
train_text = [];
validation_text = [];
test_text = [];

max_line_length = 20;
min_line_length = 4;
state_size = 100;
input_size = 100;

file_num_for_train = 1585;
file_num_for_vali = 250;
file_num_for_test = 250;
max_file_length = 0;
max_lineinfile_length = 0;

with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
# input_text = '\n'.join(doc.xpath('//content/text()'))

rawlabel = doc.xpath('//keywords/text()');



for i in range(2085):
	whichclass = 0;

	if ("technology" in rawlabel[i]):
		whichclass += 4;
	if ("entertainment" in rawlabel[i]):
		whichclass += 2;
	if ("design" in rawlabel[i]):
		whichclass += 1;


	label_vec = [0 for x in range(8)];
	label_vec[whichclass] = 1;
	if (i < 1585):
		train_label.append(label_vec);
	elif (i >= 1585 and i <1835):
		validation_label.append(label_vec);
	else:
		test_label.append(label_vec);


# In[34]:

rawtext = doc.xpath('//content/text()');
input_text = '\n'.join(doc.xpath('//content/text()'))
input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)

sentences_strings_ted = []
for line in input_text_noparens.split('\n'):
    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)

sentences_ted = []
for sent_str in sentences_strings_ted:
    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
    sentences_ted.append(tokens)

model_ted = Word2Vec(sentences_ted, size=100, window=5, min_count=5, workers=4)

def get_embedding(word):
	if (word in model_ted.wv):
		return model_ted.wv[word];
	else:
		return [0 for x in range(100)];


def get_snetences_matrix_embedding(sentences):
	return_matirx = [];
	for lines in sentences:
		for word in lines:
			return_matrix.append(get_embedding(word));
	return return_matrix;

max_length = 0;
lines = [];
seperate_files_lines = [];
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
for i in range(2085):
	tmp = re.sub(r'\([^)]*\)', '', rawtext[i]);

	sentences_strings_ted = []
	for line in tmp.split('\n'):
	    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
	    sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)

	sentences_ted = []
	for sent_str in sentences_strings_ted:
	    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
	    sentences_ted.append(tokens)

	seperate_files_lines.append(sentences_ted);
	max_file_length = max(max_file_length, len(sentences_ted));
	for _line in sentences_ted:
		max_lineinfile_length = max(max_lineinfile_length, len(_line));
		lines.append(_line)

delete_list = [];
max_length = 0;

total_words = 0;
for i in range(len(lines)):
    
	if (len(lines[i]) > max_line_length or len(lines[i])<= min_line_length):
		delete_list.append(i);
	else:
		max_length = max(max_length, len(lines[i]));
		total_words += len(lines[i])
    
for i in range(len(delete_list)):
	lines.pop(delete_list[len(delete_list)-1-i]);


print(len(lines));


# In[46]:


startnum = 1;

word_dic = {};

for line in lines:
    for word in line: 
        if word not in word_dic:
            word_dic[word] = startnum;
            startnum = startnum + 1;


## pre fetch the data
batch_size = 50;
batch_nums = len(lines) // batch_size;

embedding_label_train = np.zeros((batch_nums, batch_size, max_line_length, startnum), dtype='float32')
embedding_text_train = np.zeros((batch_nums, batch_size, max_line_length, 100), dtype='float32')
embedding_text_train_next = np.zeros((1585, max_file_length, max_lineinfile_length, 100), dtype='float32')


def generate_data(batch_size):
    lines_label = [];
    seperate_text = [];
    for i in range(batch_nums):
        seperate_text.append(lines[i*batch_size:(i+1)*batch_size]);
    bar = progressbar.ProgressBar(max_value=(len(lines)))
    bar_index = 0;
    for i in range(batch_nums):
        for j in range(batch_size):
            bar_index += 1
            bar.update(bar_index)
            for z in range(max_line_length):
                if z < len(seperate_text[i][j])-1:
                    embedding_label_train[i][j][z][word_dic[seperate_text[i][j][z+1]]] = 1;
                    embedding_text_train[i][j][z] = get_embedding(seperate_text[i][j][z]);
                elif z == len(seperate_text[i][j]) -1:
                    embedding_label_train[i][j][z][0] = 1;
                    embedding_text_train[i][j][z] = get_embedding(seperate_text[i][j][z]);
    for i in range(file_num_for_train):
    	for j in range(len(seperate_files_lines[i])):
    		for z in range(len(seperate_files_lines[i][j])):
    			embedding_text_train_next[i][j][z] = get_embedding(seperate_files_lines[i][j][z]);
generate_data(50);


# In[97]:

train_text = np.zeros([file_num_for_train, 100],dtype='float32');
validation_text = np.asarray(validation_text);
test_text = np.asarray(test_text);
train_label = np.asarray(train_label);
validation_label = np.asarray(validation_label);
test_label = np.asarray(test_label);

# print(max_length)
# print(max_lines)
state_size = 100;

X = tf.placeholder(tf.float32, [batch_size, max_line_length, 100]);
y = tf.placeholder(tf.float32, [batch_size, max_line_length, startnum]);

X_for_second_stage = tf.placeholder(tf.float32, [max_file_length, max_lineinfile_length, 100]);
init_state = tf.placeholder(tf.float32, [None, state_size]);
#init_state_second = tf.placeholder(tf.float32, [max_file_length, state_size])

rnn_inputs = tf.unstack(X, axis = 1)
rnn_inputs_for_second_stage = tf.unstack(X_for_second_stage, axis = 1)

with tf.variable_scope("rnn_cell"):
    W1 = tf.get_variable("W1", [100 + state_size, state_size],initializer=tf.constant_initializer(0.0));
    b1 = tf.get_variable("b1", [state_size], initializer=tf.constant_initializer(0.0));

def rnn_cell(rnn_input, state):
	with tf.variable_scope("rnn_cell",reuse=True):
		W1 = tf.get_variable("W1", [100+state_size, state_size]);
		b1 = tf.get_variable("b1", [state_size]);

	return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W1) + b1)

state = init_state;
state_for_next = init_state;
rnn_outputs = [];
rnn_outputs_for_second_stage = [];

for rnn_input in rnn_inputs:
	state = rnn_cell(rnn_input, state)
	rnn_outputs.append(state)

for rnn_inputu in rnn_inputs_for_second_stage:
	state_for_next = rnn_cell(rnn_inputu, state_for_next)
	rnn_outputs_for_second_stage.append(state_for_next)

final_state = rnn_outputs[-1];

with tf.variable_scope("softmax"):
	U = tf.get_variable('U', [state_size, startnum]);
	p = tf.get_variable('p', [startnum], initializer=tf.constant_initializer(0.0));

logits = [tf.matmul(rnn_output, U)+p for rnn_output in rnn_outputs];
predictions = [tf.nn.softmax(logit) for logit in logits];

correct_answer = tf.unstack(y, axis = 1);

loss = tf.reduce_mean(-tf.reduce_sum(correct_answer*tf.log(predictions), reduction_indices=[1]));
train_step = tf.train.AdagradOptimizer(0.003).minimize(loss);

represent = tf.add_n(rnn_outputs_for_second_stage)
represent_to_embedding = tf.reduce_mean(represent, axis = 0);

print(max_file_length)
def train_network(num_epochs, num_steps, state_size=100, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # training_losses = []
        
        bar_first = progressbar.ProgressBar(max_value=(len(embedding_text_train)))
       	bar_first_index = 0;
        training_state = np.zeros((batch_size, state_size))
        for train_step_num in range(1):
            for idx, epoch in enumerate(embedding_text_train):
                bar_first_index += 1;
                bar_first.update(bar_first_index);
                training_loss = 0
                tr_losses, training_state, training_state= sess.run([loss, train_step, final_state],
                                  feed_dict={X:embedding_text_train[idx], y:embedding_label_train[idx], init_state:training_state})
        training_state_second_stage = np.zeros((max_file_length, state_size))

        bar = progressbar.ProgressBar(max_value=(1586))
        bar_index = 0;
        for idx in range(file_num_for_train):
            bar_index += 1
            bar.update(bar_index);
            tmp = sess.run([represent_to_embedding], feed_dict={X_for_second_stage:embedding_text_train_next[idx], init_state:training_state_second_stage});
            print(tmp);
            train_text[idx] = tmp[0];
    return training_losses
train_network(batch_nums, 1000);


def training_network(hidden_layer_num, step_length, whether_training, whether_dropout):

	X = tf.placeholder(tf.float32, [None, 100]);
	y = tf.placeholder(tf.float32, [None, 8]);

	W = tf.Variable(tf.random_normal([100, hidden_layer_num]));
	b = tf.Variable(tf.random_normal([hidden_layer_num]));

	if (whether_dropout):
		W = tf.nn.dropout(W, 0.5);
		h = tf.nn.tanh(tf.matmul(X, W)+b);
	else:
		h = tf.nn.tanh(tf.matmul(X, W)+b);

	V = tf.Variable(tf.random_normal([hidden_layer_num, 8]));
	c = tf.Variable(tf.random_normal([8]));

	if (whether_dropout):
		V = tf.nn.dropout(V, 0.5);
		p = tf.nn.softmax(tf.matmul(h, V) + c);
	else:
		p = tf.nn.softmax(tf.matmul(h, V) + c);

	if (whether_training):
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(p), reduction_indices=[1]))
		train_step = tf.train.GradientDescentOptimizer(step_length).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(p,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
		init = tf.initialize_all_variables();
		test_feed = {X: test_text, y: test_label};
		with tf.Session() as sess:
			sess.run(init)
			cost = 10000;
			batch_start = 0;
			for step in range(100000):
				x1,y1,batch_start = get_next_batch(50, batch_start);
				sess.run(train_step, feed_dict={X: x1, y: y1});
				if ((step+1)%100==0):
					validation_feed = {X: validation_text, y: validation_label};
					curcost = sess.run(cross_entropy, feed_dict=validation_feed);
					print("step: %f" % step)
					print("cost on validation set: %f" % curcost);
					print("accuracy on validation set: %f" % accuracy.eval(validation_feed));

					print("cost on training set: %f" % sess.run(cross_entropy, feed_dict={X: train_text, y: train_label}));
					print("accuracy on training set: %f" % accuracy.eval({X: train_text, y: train_label}));
					print("accuracy on test data: %f" % sess.run(accuracy, feed_dict=test_feed));
					print(" ")
					if step == 100000-1:
						print("accuracy on test data: %f" % sess.run(accuracy, feed_dict=test_feed));
						return True;					
					else:
						cost = curcost
			return False;
training_network(100, 0.003, 1, 0);


# In[68]:




# In[ ]:



