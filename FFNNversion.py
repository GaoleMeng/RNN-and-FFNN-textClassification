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
output_notebook()

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
	whichclass = "ooo";
	if ("technology" in rawlabel[i]):
		whichclass = whichclass.replace(whichclass[0], "T", 1);
	if ("entertainment" in rawlabel[i]):
		whichclass = whichclass.replace(whichclass[1], "E", 1);
	if ("design" in rawlabel[i]):
		whichclass = whichclass.replace(whichclass[2], "D", 1);

	if (i < 1085):
		train_label.append(whichclass);
	elif (i >= 1085 and i <1835):
		validation_label.append(whichclass);
	else:
		test_label.append(whichclass);

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


print(get_embedding("fuckshit"))

for i in range(2085):

	if (i < 1085):
		train_text.append(re.sub(r'\([^)]*\)', '', rawtext[i]));
	elif (i >= 1085 and i <1835):
		validation_text.append(re.sub(r'\([^)]*\)', '', rawtext[i]));
	else:
		test_text.append(re.sub(r'\([^)]*\)', '', rawtext[i]));





#print(train_text)






del doc


# i = input_text.find("Hyowon Gweon: See this?")
# print(input_text[i-20:i+150])







