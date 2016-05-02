from postagger import history
from postagger import tagger
from postagger import decoder
from postagger import indicator
from postagger import history
import numpy as np

t=tagger.ChineseTagger()
t.readinTrainData()

with open('rule_bigram.txt','w',encoding="utf-8") as rule1out:
	bigramdict=dict()
	for item in t.trainTagList:
		for i in range(len(item)):
			if(i==0):
				newrecord="START"+"+"+item[0]
			else:
				newrecord=item[i-1]+"+"+item[i]
			if(newrecord not in bigramdict):
				bigramdict[newrecord]=0
			else:
				bigramdict[newrecord]=bigramdict[newrecord]+1
	for item in bigramdict:
		if(bigramdict[item]>5):
			biTagUnit=item.split("+")
			rule1out.write("\t%s\t\t\t\t\t\t%s\n" % (biTagUnit[0],biTagUnit[1]))
