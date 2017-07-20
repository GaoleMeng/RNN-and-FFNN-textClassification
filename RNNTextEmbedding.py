
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

max_line_length = 10;
min_line_length = 3;
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
	if (i < file_num_for_train):
		train_label.append(label_vec);
	elif (i >= file_num_for_train and i < file_num_for_train+file_num_for_vali):
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
embedding_text_train_next = np.zeros((file_num_for_train, max_file_length, max_lineinfile_length, 100), dtype='float32')
embedding_text_vali_next = np.zeros((file_num_for_vali, max_file_length, max_lineinfile_length, 100), dtype='float32')
embedding_text_test_next = np.zeros((file_num_for_test, max_file_length, max_lineinfile_length, 100), dtype='float32')

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
    for i in range(file_num_for_test+file_num_for_vali+file_num_for_train):
        for j in range(len(seperate_files_lines[i])):
            for z in range(len(seperate_files_lines[i][j])):
                if (i < file_num_for_train):
                    embedding_text_train_next[i][j][z] = get_embedding(seperate_files_lines[i][j][z]);
                elif (i >= file_num_for_train and i < (file_num_for_train+file_num_for_vali)):
                	embedding_text_vali_next[i- file_num_for_train][j][z] = get_embedding(seperate_files_lines[i][j][z]);
                else:
                    embedding_text_test_next[i- file_num_for_train - file_num_for_vali][j][z] = get_embedding(seperate_files_lines[i][j][z]);
generate_data(50);


state_size = 100;

X = tf.placeholder(tf.float32, [batch_size, max_line_length, 100]);
y = tf.placeholder(tf.float32, [batch_size, max_line_length, startnum]);

X_for_second_stage = tf.placeholder(tf.float32, [max_file_length, max_lineinfile_length, 100]);
init_state = tf.placeholder(tf.float32, [None, state_size]);
#init_state_second = tf.placeholder(tf.float32, [max_file_length, state_size])

rnn_inputs = tf.unstack(X, axis = 1)
rnn_inputs_for_second_stage = tf.unstack(X_for_second_stage, axis = 1)

with tf.variable_scope("rnn_cell"):
    W1 = tf.get_variable("W1", [100 + state_size, state_size]);
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

        bar = progressbar.ProgressBar(max_value=(file_num_for_vali+file_num_for_train+file_num_for_test+1))
        bar_index = 0;

        text_embedding_train = open("data/textEmbeddingTrain.txt","w")
        text_embedding_vali = open("data/textEmbeddingVali.txt","w")
        text_embedding_test = open("data/textEmbeddingtest.txt","w")
        for idx in range(file_num_for_train+file_num_for_test+file_num_for_vali):
            bar_index += 1
            bar.update(bar_index);
            ostr = "";
            if idx < file_num_for_train:
                tmp = sess.run([represent_to_embedding], feed_dict={X_for_second_stage:embedding_text_train_next[idx], init_state:training_state_second_stage});
                for i_ou in range(len(tmp[0])):
                	ostr = ostr + " " + str(tmp[0][i_ou]);
                text_embedding_train.write(ostr+"\n");
            elif idx >= file_num_for_train and idx < (file_num_for_vali+file_num_for_train):
                tmp = sess.run([represent_to_embedding], feed_dict={X_for_second_stage:embedding_text_vali_next[idx], init_state:training_state_second_stage});
                for i_ou in range(len(tmp[0])):
                	ostr = ostr + " " + str(tmp[0][i_ou]);

                text_embedding_vali.write(ostr+"\n");
            else:
                tmp = sess.run([represent_to_embedding], feed_dict={X_for_second_stage:embedding_text_test_next[idx], init_state:training_state_second_stage});
                for i_ou in range(len(tmp[0])):
                	ostr = ostr + " " + str(tmp[0][i_ou]);
                text_embedding_test.write(ostr+"\n");
    return []
train_network(batch_nums, 1000);




