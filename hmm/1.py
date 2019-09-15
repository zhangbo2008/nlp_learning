

### const.py



#%%

# 未登录词
UNK = None
# 句子开始标记，代表句子的开头
START_TOKEN = '<s>'
# 句子结束标记，代表句子的结尾
END_TOKEN = '</s>'

#%% md

### processing.py



#%%

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@description: 句子的处理，字典的构建
@author: Sean QQ: 929325776
'''

import const

#加入起始标记        给标记前后加入标签.
def build_tags(tags):
	out = []
	for sentence in tags:
		items = [x.lower() for x in sentence]
		items.insert(0, const.START_TOKEN)
		items.append(const.END_TOKEN)
		out.append(items)
	return out

# 构建ungram词频词典
def build_undict(tags): #词频字典
	undict = {}
	for items in tags:
		for word in items:
			if word == const.START_TOKEN or word == const.END_TOKEN:
				continue
			if word not in undict:
				undict[word] = 1
			else:
				undict[word] += 1
	return undict


# 构建bigram词频词典，其中以三元组(u, v)作为词典的键
def build_bidict(tags):
	bidict = {}
	for items in tags:
		for i in range(len(items)-1):
			tup = (items[i], items[i+1])
			if tup not in bidict:
				bidict[tup] = 1
			else:
				bidict[tup] += 1
	return bidict

# 构建trigram词频词典，其中以三元组(u, v, w)作为词典的键
def build_tridict(tags):
	tridict = {}
	for items in tags:
		items.insert(0, const.START_TOKEN)
		for i in range(len(items) -2):
			tup = (items[i], items[i+1], items[i+2])
			if tup not in tridict:
				tridict[tup] = 1
			else:
				tridict[tup] += 1
	return tridict

# 构建(词,词性)词频字典，以及统计词频
def build_count_dict(datas, tags):
	tagword_dict = {}
	wordcount = {}
	tagcount = {}
	for i, data in enumerate(datas):
		tag = tags[i][1:-1]
		for idx, d in enumerate(data):
			tup = (tag[idx], d)
			if tup not in tagword_dict:
				tagword_dict[tup] = 1
			else:
				tagword_dict[tup] += 1

			if d not in wordcount:
				wordcount[d] = 1
			else:
				wordcount[d] += 1
			if tag[idx] not in tagcount:
				tagcount[tag[idx]] = 1
			else:
				tagcount[tag[idx]] += 1
	return tagword_dict, wordcount, tagcount

#%% md

### hmm.py



#%%

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@description: bigram hmm, trigram hmm, _viterbi
@author: Sean QQ: 929325776
'''

import math
import const
from processing import *

'''bigram hmm'''
class BiHMM(object):
	def __init__(self, datas, tags):
		self.datas = datas
		self.tags = build_tags(tags)
		self.undict = build_undict(self.tags)
		self.bidict = build_bidict(self.tags)
		self.tagword, self.wordcount, self.tagcount = build_count_dict(datas, self.tags)
		self.postags = [tag for tag in self.undict]

	def calc_e_prob(self, *args):
		if len(args) != 2:
			raise ValueError('two tags is required')

		n = 0.0
		m = 0.0
		if args in self.tagword:
			n = self.tagword[args]
		if args[0] in self.undict:
			m = self.undict[args[0]]
		return (n + 1) * 1.0 / (m + len(self.wordcount)*len(self.undict))

	def calc_prob(self, *args):
		if len(args) != 2:
			raise ValueError('two tags is required')

		n = 0.0
		m = 0.0
		if args in self.bidict:
			n = self.bidict[args]
		if args[0] in self.undict:
			m = self.undict[args[0]]
		return (n + 1) * 1.0 / (m + len(self.postags)**2)

	def calc_tags_prob(self, tags):
		prob = 0
		prev_tag = const.START_TOKEN
		for tag in tags:
			tag_prob = self.calc_prob(prev_tag, tag)
			prob += tag_prob
			prev_tag = tag
		return prob

	def calc_tagword_proba(self, tag, word):
		prob = 0.0
		tagword = (tag, word)
		if tagword in self.tagword:
			prob = float(self.tagword[tagword]) / self.tagcount[tag]
		return prob

	# @param vb _viterbi
	def pred(self, sentence, vb=False):
		if vb:
			# _viterbi
			return self._viterbi(sentence)

		wordtag = []
		max_prob = 0.0
		max_tag = None
		#total_prob = None
		for word in sentence:
			for tag1 in self.postags:
				for tag2 in self.postags:
					q = self.calc_tags_prob((tag1, tag2))
					e = self.calc_tagword_proba(tag2, word)
					prob = q*e*1.0
					if prob >= max_prob:
						max_prob = prob
						max_tag = tag2
			wordtag.append((word, max_tag))
			'''
			if total_prob == None:
				total_prob = max_prob
			else:
				total_prob *= max_prob 
			'''
			max_prob = 0.0
		return wordtag


	def _viterbi_decode(self, sentence, score, trace):
		result = []
		tmp = -float('inf')
		res_x = 0
		for idx, val in enumerate(self.postags):
			if tmp < score[idx][len(sentence)-1]:
				tmp = score[idx][len(sentence)-1]
				res_x = idx
		result.append(res_x)
		for idx in range(len(sentence)-1, 0, -1):
			result.append(trace[result[-1]][idx])
		result.reverse()
		result_pos = []
		result_pos = [self.postags[k] for k in result]
		wordtag = list(zip(sentence, result_pos))
		return wordtag

	def _viterbi(self, sentence):
		row = len(self.postags)
		col = len(sentence)

		trace = [[-1 for i in range(col)] for i in range(row)]
		score = [[-1 for i in range(col)] for i in range(row)]

		for idx, val in enumerate(sentence):
			if idx == 0:
				for idx_pos, val_pos in enumerate(self.postags):
					score[idx_pos][idx] = self.calc_e_prob(val_pos, sentence[idx]) # emit
			else:
				for idx_pos, val_pos in enumerate(self.postags):
					tmp = -float('inf')
					trace_tmp = -1
					for idx_pos2, val_pos2 in enumerate(self.postags):
						r = score[idx_pos2][idx-1]*self.calc_prob(val_pos2, val_pos)
						if r > tmp:
							tmp = r
							trace_tmp = idx_pos2
						trace[idx_pos][idx] = trace_tmp
						score[idx_pos][idx] = tmp*self.calc_e_prob(val_pos, val)
		return self._viterbi_decode(sentence, score, trace)

class TriHMM(BiHMM):
	def __init__(self, datas, tags):
		BiHMM.__init__(self, datas, tags)
		self.tridict = build_tridict(self.tags)

	def calc_prob(self, *args):
		if len(args) != 3:
			raise ValueError('three tags is required')

		n = 0.0
		m = 0.0
		bitup = (args[0], args[1])
		if args in self.tridict:
			n = self.tridict[args]
		if bitup in self.bidict:
			m = self.bidict[bitup]
		return (n + 1) * 1.0 / (m + len(self.postags)**2)


		prob = 0
		if self.smooth != None:
			prob = self.smooth(args[0], args[1], args[2], tridict=self.tridict, bidict=self.bidict, undict=self.undict)
		else:
			bitup = (args[0], args[1])
			if args in self.tridict and bitup in self.bidict:
				return float(self.tridict[args]) / self.bidict[bitup]
		return prob

	def calc_tags_prob(self, tags):
		prob = 0
		prev_stack = [const.START_TOKEN, const.START_TOKEN]
		for tag in tags:
			tag_prob = self.calc_prob(prev_stack[0], prev_stack[1], tag)
			prob += tag_prob
			prev_stack[0] = prev_stack[1]
			prev_stack[1] = tag
		return prob

	# @param vb _viterbi
	def pred(self, sentence, vb=False):
		if vb:
			return self._viterbi(sentence)
		wordtag = []
		max_prob = 0.0
		max_tag = None
		#total_prob = None
		for word in sentence:
			for tag1 in self.postags:
				for tag2 in self.postags:
					for tag3 in self.postags:
						q = self.calc_tags_prob((tag1, tag2, tag3))
						e = self.calc_tagword_proba(tag3, word)
						prob = q*e*1.0
						if prob >= max_prob:
							max_prob = prob
							max_tag = tag3
			wordtag.append((word, max_tag))
			'''
			if total_prob == None:
				total_prob = max_prob
			else:
				total_prob *= max_prob 
			'''
			max_prob = 0.0
		return wordtag

	def _viterbi_decode(self, sentence, score, trace):
		result = []
		tmp = -float('inf')
		res_x = 0
		res_y = 0
		for idx, val in enumerate(self.postags):
			for idx_pos2, val_pos2 in enumerate(self.postags):
				if tmp < score[idx_pos2][idx][len(sentence)-1]:
					tmp = score[idx_pos2][idx][len(sentence)-1]
					res_x = idx
					res_y = idx_pos2
		result.extend([res_x, res_y])
		for idx in range(len(sentence)-1, 0, -1):
			result.append(trace[result[-2]][result[-1]][idx])
		result.reverse()
		result_pos = []
		result_pos = [self.postags[k] for k in result]
		wordtag = list(zip(sentence, result_pos))
		return wordtag

	def _viterbi(self, sentence):
		row = len(self.postags)
		col = len(sentence)

		trace = [[[-1 for i in range(col)] for i in range(row)] for i in range(row)]
		score = [[[-1 for i in range(col)] for i in range(row)] for i in range(row)]

		for idx, val in enumerate(sentence):
			if idx == 0:
				for idx_pos, val_pos in enumerate(self.postags):
					score[idx_pos][0][idx] = self.calc_e_prob(val_pos, sentence[idx]) # emit
			else:
				for idx_pos, val_pos in enumerate(self.postags):
					tmp = -float('inf')
					trace_tmp = -1
					for idx_pos2, val_pos2 in enumerate(self.postags):
						for idx_pos3, val_pos3 in enumerate(self.postags):
							r = score[idx_pos3][idx_pos2][idx-1]*self.calc_prob(val_pos3, val_pos2 ,val_pos)
							if r > tmp:
								tmp = r
								trace_tmp = idx_pos3
							trace[idx_pos][idx_pos2][idx] = trace_tmp
							score[idx_pos][idx_pos2][idx] = tmp*self.calc_e_prob(val_pos, val)
		return self._viterbi_decode(sentence, score, trace)

#%% md

## 结果分析

#%% md

** bigram hmm **

bigram hmm

[('小明', 'nr'), ('爱', 'v'), ('老鼠', 'n'), ('和', 'c'), ('狗', 'n')]

bigram hmm with viterbi decode

[('小明', 'nr'), ('爱', 'v'), ('老鼠', 'n'), ('和', 'v'), ('狗', 'n')]

**trigram hmm**

trigram hmm

[('小明', 'nr'), ('爱', 'v'), ('老鼠', 'n'), ('和', 'c'), ('狗', 'n')]

trigram hmm with viterbi decode

[('小明', 'nr'), ('爱', 'v'), ('老鼠', 'n'), ('和', 'c'), ('狗', 'n')]


#%% md

## 项目后续

过段时间会加入深度学习在NLP上的应用，如果你感兴趣，可以关注我的公众号，或者star, watch 本项目哦

#%% md

## 联系作者

@author sean

@qq 929325776

有什么问题，可以联系我，一起讨论
