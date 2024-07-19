from typing import List
import flair
from flair.data import Dictionary, Sentence, Token, Label
#from flair.datasets import CONLL_03, CONLL_03_DUTCH, CONLL_03_SPANISH, CONLL_03_GERMAN
import flair.datasets as datasets
from flair.data import MultiCorpus, Corpus
from flair.list_data import ListCorpus
import flair.embeddings as Embeddings
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter
# initialize sequence tagger
from flair.models import SequenceTagger
from pathlib import Path
import argparse
import yaml
from flair.utils.from_params import Params
# from flair.trainers import ModelTrainer
# from flair.trainers import ModelDistiller
# from flair.trainers import ModelFinetuner
from flair.config_parser import ConfigParser
import pdb
import sys
import os
import logging
from pathlib import Path
import torch.nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.autograd as autograd
import flair.nn
import torch
from flair.custom_data_loader import ColumnDataLoader
from flair.datasets import DataLoader
import time,flask,json
from flask import request, jsonify
keep_order = False
def count_parameters(model):
        import numpy as np
        total_param = 0
        for name,param in model.named_parameters():
                num_param = np.prod(param.size())
                # print(name,num_param)
                total_param+=num_param
        return total_param


log = logging.getLogger("flair")
config = Params.from_file("config/xlmr-large-first_10epoch_1batch_4accumulate_0.000005lr_10000lrrate_en_monolingual_crf_fast_norelearn_sentbatch_sentloss_nodev_finetune_twitter15_doc_joint_multiview_posterior_4temperature_captionobj_classattr_vinvl_ocr_ner24.yaml")
# pdb.set_trace()
config = ConfigParser(config,all=False,zero_shot=False,other_shot=False,predict=False
                                          , save_embedding = False)
# pdb.set_trace()

student=config.create_student(nocrf=False)
log.info(f"Model Size: {count_parameters(student)}")
corpus=config.corpus

teacher_func=config.create_teachers
if 'is_teacher_list' in config.config:
	if config.config['is_teacher_list']:
		teacher_func=config.create_teachers_list

# pdb.set_trace()
if 'trainer' in config.config:
	trainer_name=config.config['trainer']
else:
	if 'ModelDistiller' in config.config:
		trainer_name='ModelDistiller'
	elif 'ModelFinetuner' in config.config:
		trainer_name='ModelFinetuner'
	elif 'ReinforcementTrainer' in config.config:
		trainer_name='ReinforcementTrainer'
	else:
		trainer_name='ModelDistiller'


eval_mini_batch_size = int(config.config['train']['mini_batch_size'])

teacher_func=config.create_teachers
print('Batch Size:',eval_mini_batch_size)
base_path=Path(config.config['target_dir'])/config.config['model_name']
	# 加载模型
if (base_path / "best-model.pt").exists():
	print('Loading pretraining best model')
	if trainer_name == 'ReinforcementTrainer':
		student = student.load(base_path / "best-model.pt", device='cpu')
		for name, module in student.named_modules():
			if 'embeddings' in name or name == '':
				continue
			else:
				module.to(flair.device)
		for name, module in student.named_parameters():
			module.to(flair.device)
	else:
		student = student.load(base_path / "best-model.pt")
	
elif (base_path / "final-model.pt").exists():
	print('Loading pretraining final model')
	student = student.load(base_path / "final-model.pt")
else:
	assert 0, str(base_path)+ ' not exist!'
#加载模型结束
for embedding in student.embeddings.embeddings:
		# manually fix the bug for the tokenizer becoming None
		if hasattr(embedding,'tokenizer') and embedding.tokenizer is None:
			from transformers import AutoTokenizer
			name = embedding.name
			if '_v2doc' in name:
				name = name.replace('_v2doc','')
			if '_extdoc' in name:
				name = name.replace('_extdoc','')
			embedding.tokenizer = AutoTokenizer.from_pretrained(name)
		if hasattr(embedding,'model') and hasattr(embedding.model,'encoder') and not hasattr(embedding.model.encoder,'config'):
			embedding.model.encoder.config = embedding.model.config
if not hasattr(student,'use_bert'):
		student.use_bert=False
if hasattr(student,'word_map'):
	word_map = student.word_map
else:
	word_map = None
if hasattr(student,'char_map'):
	char_map = student.char_map
else:
    char_map = None
print(type(corpus.train))
def makePrediction(sentence_requseted):
	sentence_input = Sentence(sentence_requseted,use_tokenizer=True)
	loader=ColumnDataLoader([sentence_input],eval_mini_batch_size,use_bert=student.use_bert, model = student, sort_data = not keep_order,
							sentence_level_batch = config.config[trainer_name]['sentence_level_batch'] if 'sentence_level_batch' in config.config[trainer_name] else True)
	loader.assign_tags(student.tag_type,student.tag_dictionary)
	lines: List[str] = []
	# train_eval_result, train_loss = student.evaluate(loader,out_path=Path('outputs/train'+'.'+corpus.targets[0]+'.conllu'),embeddings_storage_mode="none",prediction_mode=True)
	# if train_eval_result is not None:
	# 	print('Current accuracy: ' + str(train_eval_result.main_score*100))
	# 	print(train_eval_result.detailed_results)
	with torch.no_grad():
				batch_no: int = 0
				for batch in loader:
					batch_no += 1
					with torch.no_grad():
						# pdb.set_trace()
						features = student.forward(batch,prediction_mode=True)
						mask=student.mask
						tags, _ = student._obtain_labels(features, batch)
						for (sentence, sent_tags) in zip(batch, tags):
							for (token, tag) in zip(sentence.tokens, sent_tags):
								token: Token = token
								token.add_tag_label("predicted", tag)

								# append both to file for evaluation
								eval_line = "{} {} {}".format(
									token.text,
									tag.value,
									tag.score,
								)
								lines.append(eval_line)
	return lines


app = flask.Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
	data = request.form.get('sentence')
	input_query = data['sentence']
	lines = makePrediction(input_query)
	res  ={'Tokens':[] , 'Entity_type':[],'Score':[]}
	for line in lines:
		temp = line.split()
		res['Tokens'].append(temp[0])
		res['Entity_type'].append(temp[1])
		res['Score'].append(temp[2])
	return json.dumps(res, ensure_ascii=False)
if __name__  == '__main__':
	app.run(host=0.0.0.0, port=5000)
	
