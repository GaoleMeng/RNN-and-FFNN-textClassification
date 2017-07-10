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
	print(label_vec);
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
	sum_embedding = np.asarray([0 for x in range(100)])

	for t in range(len(sentences_ted)):
		for p in range(len(sentences_ted[t])):
			sum_embedding = sum_embedding + np.asarray(get_embedding(sentences_ted[t][p]));
	# sum_embedding = np.asarray(sum_embedding);

	if (i < 1085):
		train_text.append(sum_embedding);
	elif (i >= 1085 and i <1835):
		validation_text.append(sum_embedding);
	else:
		test_text.append(sum_embedding);


train_text = np.asarray(train_text);
validation_text = np.asarray(validation_text);
test_text = np.asarray(test_text)

def training_network(hidden_layer_num, step_length, whether_training):

	X = tf.placeholder(tf.float32, [None, 100]);
	y = tf.placeholder(tf.float32, [None, 8]);

	W = tf.Variable(tf.zeros([100, hidden_layer_num]));
	b = tf.Variable(tf.zeros([hidden_layer_num]));

	h = tf.nn.tanh(tf.matmul(X, W)+b);

	V = tf.Variable(tf.zeros([hidden_layer_num, 8]));
	c = tf.Variable(tf.zeros([8]));

	p = tf.nn.softmax(tf.matmul(h, V) + c);

	if (whether_training):
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(p), reduction_indices=[1]))
		train_step = tf.train.GradientDescentOptimizer(step_length).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(p,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
		init = tf.initialize_all_variables();

		with tf.Session() as sess:
			sess.run(init)
			cost = 10000;
			for step in range(200000):
				sess.run(train_step, feed_dict={X: train_text, y: train_label});
				if ((step+1)%100==0):
					validation_feed = {X: validation_text, y: validation_label};
					curcost = sess.run(cross_entropy, feed_dict=validation_feed);
					print("cost on validation set: %f" % curcost);
					print("accuracy on validation set: %f" % sess.run(accuracy, feed_dict=validation_feed));
					print("cost on training set: %f" % sess.run(cross_entropy, feed_dict={X: train_text, y: train_label}));
					print("accuracy on training set: %f" % sess.run(accuracy, feed_dict={X: train_text, y: train_label}));
					if step == 200000-1:
						test_feed = {X: test_text, y: test_label};
						print("accuracy on test data: %f" % sess.run(accuracy, feed_dict=test_feed));
						return True;					
					else:
						cost = curcost
					


			return False;
training_network(100, 0.1, 1);







#print(train_text)






del doc


# i = input_text.find("Hyowon Gweon: See this?")
# print(input_text[i-20:i+150])







