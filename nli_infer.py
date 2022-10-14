import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import RobertaTokenizer
import sys,getopt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from make_cf import make_confusion_matrix

def construct_pipeline(model_name):
	model = AutoModelForSequenceClassification.from_pretrained(model_name)
	# tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer = AutoTokenizer.from_pretrained('roberta-base')
	# tokenizer = RobertaTokenizer.from_pretrained(model_name)
	pipe = pipeline('text-classification',model=model,tokenizer= tokenizer)
	return pipe

def read_input(input_file):
	df  = pd.read_csv(input_file)
	sentence1 = list(df['premise'])
	sentence2 = list(df['hypothesis'])
	labels = list(df['final-label'])
	return df,sentence1, sentence2, labels

def construct_input(model, sentence1, sentence2):
	# print(sentence1, sentence2)
	if 'roberta' in model:
		return construct_input_roberta(sentence1,sentence2)
	elif 'deberta' in model:
		return construct_input_deberta(sentence1, sentence2)	
	else:
		# return construct_input_roberta(sentence1,sentence2)
		return construct_input_deberta(sentence1, sentence2)	
		# return construct_input_no_token(sentence1,sentence2)

def construct_input_roberta(sentence1, sentence2):
	return sentence1+'. '+sentence2

def construct_input_deberta(sentence1, sentence2):
	return '[CLS] '+ sentence1+' [SEP] ' + sentence2 + ' [SEP]'

def construct_input_no_token(sentence1, sentence2):
	return '<s>' +  sentence1 + '</s></s> ' + sentence2 + '</s>'


def run_pipe(model_name,pipe, sentence1, sentence2):
	predicts = []
	for i,sent in enumerate(sentence1):
		input_pair = construct_input(model_name,sent,sentence2[i])
		# print(sent)
		# print(sentence2[i])
		# print(input_pair)
		res = pipe(input_pair)
		predict_label = res[0]['label'].lower()
		# print(predict_label)
		#textattack version
		# if predict_label == 'label_0':
		# 	predict_label = 'contradiction'
		# elif predict_label == 'label_1':
		# 	predict_label = 'neutral'
		# elif predict_label == 'label_2':
		# 	predict_label = 'entailment'
		#base version
		if predict_label == 'label_0':
			predict_label = 'entailment'
		elif predict_label == 'label_1':
			predict_label = 'neutral'
		elif predict_label == 'label_2':
			predict_label = 'contradiction'
		#alphabet version
		# if predict_label == 'label_0':
		# 	predict_label = 'contradiction'
		# elif predict_label == 'label_1':
		# 	predict_label = 'entailment'
		# elif predict_label == 'label_2':
		# 	predict_label = 'neutral'

		predicts.append(predict_label)
	return predicts
	
def evaluate(predicts, labels):
	labels_binary = []
	predicts_binary = []
	for predict in predicts:
		if predict != 'entailment':
			predict = 'not_entailment'
		predicts_binary.append(predict)


	for label in labels:
		if label != 'entailment':
			label = 'not_entailment'
		labels_binary.append(label)

	# print(model_name)
	print('Accuracy', accuracy_score(labels, predicts))
	print('Accuracy Binary', accuracy_score(labels_binary, predicts_binary))
	print(classification_report(labels,predicts,digits=4))
	print(classification_report(labels_binary,predicts_binary,digits=4))

	target_names = ['contradiction', "entailment", 'neutral']
	cm = confusion_matrix(labels, predicts, normalize = "true", labels = target_names)
	# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = target_names).plot(cmap = 'Blues', colorbar="false")
	# make_confusion_matrix(cm, 
        #               categories=target_names, 
        #               cmap='binary')
	plt.savefig('nan')
	print(cm)
	# print(cm.diagonal())


def highlight_difference(row):
    if row['label'] != row['predict']:
        color = 'red'

    background = ['background-color: {}'.format(color) for _ in row]
    return background


# def infer(sentence1, sentence2):
# 	predicts = []
# 	for i,sent in enumerate(sentence1):
# 		input_pair = construct_input_deberta(sent,sentence2[i])
# 		res = pipe(input_pair)
# 		predict_label = res[0]['label'].lower()
# 		if predict_label == 'label_0':
# 			predict_label = 'entailment'
# 		elif predict_label == 'label_1':
# 			predict_label = 'neutral'
# 		elif predict_label == 'label_2':
# 			predict_label = 'contradiction'
# 		predicts.append(predict_label)
# 	return predicts


def main(argv):
	inputfile = ''
	outputfile = ''
	model_name = ''
	try:
		opts, args = getopt.getopt(argv,"hi:o:m:",["ifile=","ofile="])
	except getopt.GetoptError:
		print('test.py -i <inputfile> -o <outputfile>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('nli_infer.py -i <inputfile> -m <model_name_or_path>')
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-m", "--model_path"):
			model_name = arg
		# elif opt in ("-o", "--output_dir"):
		# 	outputfile = arg	
	print('Input file:', inputfile)
	# print('Output dir:', outputfile)
	print('Model:', model_name)

	df, sentence1, sentence2, labels = read_input(inputfile)
	pipe = construct_pipeline(model_name)

	predicts = run_pipe(model_name,pipe, sentence1, sentence2)

	df['predict'] = predicts
	df_out = pd.DataFrame(data = {'premise': sentence1, 'hypothesis': sentence2, 'label': labels, 'predict': predicts})
	evaluate(predicts, labels)
	df_out.style.apply(highlight_difference, axis = 1)
	model_name = model_name.replace('/','-')
	output_file = inputfile+'-pred-'+model_name+'.csv'
	df_out.to_csv(output_file, index = None)


if __name__ == "__main__":
   main(sys.argv[1:])
