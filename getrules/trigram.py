from postagger import history
from postagger import tagger
from postagger import decoder
from postagger import indicator
from postagger import history
import numpy as np

t=tagger.ChineseTagger()
t.readinTrainData()

with open('rule_trigram.txt','w',encoding="utf-8") as rule2out:
	trigramdict=dict()
	for item in t.trainTagList:
		for i in range(len(item)):
			if(i==0):
				newrecord="START"+"+"+"START"+"+"+item[0]
			elif(i==1):
				newrecord="START"+"+"+item[0]+"+"+item[1]
			else:
				newrecord=item[i-2]+"+"+item[i-1]+"+"+item[i]
			if(newrecord not in trigramdict):
				trigramdict[newrecord]=0
			else:
				trigramdict[newrecord]=trigramdict[newrecord]+1
	for item in trigramdict:
		if(trigramdict[item]>5):
			triTagUnit=item.split("+")
			rule2out.write("\t%s\t%s\t\t\t\t\t%s\n" % (triTagUnit[0],triTagUnit[1],triTagUnit[2]))