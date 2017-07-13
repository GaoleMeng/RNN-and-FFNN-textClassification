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
import lxml.etree


# Download the dataset if it's not already there: this may take a minute as it is 75MB
if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")


# For now, we're only interested in the subtitle text, so let's extract that from the XML:

train_label = [];
validation_label = [];
test_label = [];
train_text = [];
validation_text = [];
test_text = [];

state_size = 100;
input_size = 100;

with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
# input_text = '\n'.join(doc.xpath('//content/text()'))

rawlabel = doc.xpath('//keywords/text()');
for i in range(2085):
	# whichclass = "ooo";

	# if ("technology" in rawlabel[i]):
	# 	whichclass = whichclass.replace(whichclass[0], "T", 1);
	# if ("entertainment" in rawlabel[i]):
	# 	whichclass = whichclass.replace(whichclass[1], "E", 1);
	# if ("design" in rawlabel[i]):
	# 	whichclass = whichclass.replace(whichclass[2], "D", 1);
		# whichclass = "ooo";

	whichclass = 0;

	if ("technology" in rawlabel[i]):
		whichclass += 4;
	if ("entertainment" in rawlabel[i]):
		whichclass += 2;
	if ("design" in rawlabel[i]):
		whichclass += 1;


	label_vec = [0 for x in range(8)];
	label_vec[whichclass] = 1;
	if (i < 1085):
		train_label.append(label_vec);
	elif (i >= 1085 and i <1835):
		validation_label.append(label_vec);
	else:
		test_label.append(label_vec);

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

	for _line in sentences_ted:
		lines.append(_line)
	# X = tf.placeholder(tf.float32, [None, 100]);
	# y = tf.placeholder(tf)

	# sum_embedding = np.asarray([0 for x in range(100)])
	# num = 1
	# for t in range(len(sentences_ted)):
	# 	for p in range(len(sentences_ted[t])):
	# 		sum_embedding = sum_embedding + np.asarray(get_embedding(sentences_ted[t][p]));
	# 		num = num + 1;
	# sum_embedding = np.asarray(sum_embedding);
	# for tt in range(100):
	# 	sum_embedding[tt] /= float(num);
	# if (i < 1085):
	# 	train_text.append(sum_embedding);
	# elif (i >= 1085 and i <1835):
	# 	validation_text.append(sum_embedding);
	# else:
	# 	test_text.append(sum_embedding);


print(len(lines));
delete_list = [];
max_length = 0;
for i in range(len(lines)):
	if (len(lines[i]) > 100 or len(lines[i])<=1):
		delete_list.append(i);
	else:
		max_length = max(max_length, len(lines[i]));

print(len(delete_list))
for i in range(len(delete_list)):
	lines.pop(delete_list[len(delete_list)-1-i]);
print(len(lines))
print(max_length)

embedding_text_train = [];
embedding_label_train = [];


startnum = 1

def generate_data(batch_size):
	batch_nums = len(lines) // batch_size;

	lines_label = [];
	global startnum;

	seperate_text = [];
	for i in range(batch_nums):
		seperate_text.append(lines[i*batch_size:(i+1)*batch_size]);
	word_dic = {};
	
	for line in lines:
		for word in line:
			if word not in word_dic:
				word_dic[word] = startnum;
				startnum = startnum + 1;
	print(startnum);
	print("copy compeleted")
	for block in seperate_text:

		labelblock = [];
		print("next block")
		print(len(block))
		max_length_per_block = 0;
		for line in block:
			max_length_per_block = max(max_length_per_block, len(line));
		print(str(max_length_per_block)+"\n")

		for line in block:
			labelline = [];
			for i in range(len(line)):
				tmp_vec = [0 for x in range(startnum)];
				if i < len(line)-1:
					tmp_vec[word_dic[line[i+1]]] = 1;
				else:
					tmp_vec[0] = 1;
				labelline.append(tmp_vec);
			for i in range(max_length_per_block - len(line)):
				labelline.append([0 for x in range(startnum)])
			labelblock.append(labelline); 


	for block in seperate_text:
		labelblock = [];
		print("next block")
		max_length_per_block = 0;
		for line in block:
			max_length_per_block = max(max_length_per_block, len(line));

		for line in block:
			for i in range(len(line)):
				line[i] = get_embedding(line[i]);
			for i in range(max_length_per_block - len(line)):
				line.append([0 for x in range(100)])


# def get_next_batch_lines(batch_num):
print(startnum);
generate_data(50);

train_text = np.asarray(train_text);
validation_text = np.asarray(validation_text);
test_text = np.asarray(test_text);
train_label = np.asarray(train_label);
validation_label = np.asarray(validation_label);
test_label = np.asarray(test_label);

# print(max_length)
# print(max_lines)
state_size = 100;

X = tf.placeholder(tf.float32, [50, None, 100]);
y = tf.placeholder(tf.float32, [50, None, startnum]);
init_state = tf.zeros([50, state_size]);

rnn_inputs = tf.unstack(X, axis = 1)



with tf.variable_scope("run_cell"):
	W = tf.get_variable("W", [100 + state_size, state_size]);
	b = tf.get_variable("b", [state_size], initializer=tf.constant_initializer(0.0));

def rnn_cell(rnn_input, state):
	with tf.variable_scope("rnn_cell", reuse=True):
		W = tf.get_Variable("W", [input_size+state_size, state_size]);
		b = tf.get_variable("b", [state_size]);

	return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)


state = init_state;
rnn_outputs = [];

for rnn_input in rnn_inputs:
	state = rnn_cell(rnn_input, state)
	rnn_outputs.append(state)


with tf.variable_scope("softmax"):
	U = tf.get_Variable('U', [state_size, startnum]);
	p = tf.get_Variable('p', [startnum], initializer=tf.constant_initializer(0.0));

logits = [tf.matmul(rnn_output, U)+p for rnn_output in rnn_outputs];
predictions = [tf.nn.softmax(logit) for logit in logits];

correct_answer = tf.unstack(y, axis = 1);

loss = tf.reduce_mean(-tf.reduce_sum(correct_answer*log(predictions), reduction_indices=[1]));
train_step = tf.train.AdagradOptimizer(0.003).minimize(losses);


# with tf.Session() as sess:
# 	 sess.run(tf.global_variables_initializer())
# 	 for i, epoch in enumerate(embedding_text_train)




def get_next_batch(batch_num, batch_start):
	new_batch_start = (batch_start + batch_num - 1) % len(train_text);
	if new_batch_start > batch_start:
		tmp = batch_start;
		batch_start = (new_batch_start + 1) % len(train_text);
		return train_text[tmp:new_batch_start], train_label[tmp:new_batch_start], batch_start;
	else:
		tmp = batch_start;
		batch_start = (new_batch_start + 1) % len(train_text);

		return_train_text = np.concatenate((train_text[tmp:(len(train_text)-1)], train_text[0:batch_start]));
		return_train_label = np.concatenate((train_label[tmp:(len(train_label)-1)],train_label[0:batch_start]))
		# return_train_text = train_text[tmp:(len(train_text)-1)].append(train_text[0:batch_start]);
		# return_train_label = train_label[tmp:(len(train_label)-1)].append(train_label[0:batch_start]);
		return return_train_text, return_train_label, batch_start;


# def get_text_embedding_rnn():

# 	for i in range(len(sentences)):

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
			for step in range(1000000):
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
					if step == 1000000-1:
						print("accuracy on test data: %f" % sess.run(accuracy, feed_dict=test_feed));
						return True;					
					else:
						cost = curcost
			return False;
# training_network(50, 0.003, 1, 0);





#print(train_text)






del doc





