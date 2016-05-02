from postagger import history
from postagger import tagger
from postagger import decoder
from postagger import indicator
from postagger import history
import numpy as np

t=tagger.ChineseTagger()
t.readinTrainData()

with open('rule_nextwordNS.txt','w',encoding="utf-8") as rule4out:
	nnsgramdict=dict()
	for i in range(len(t.trainTagList)):
		for j in range(len(t.trainTagList[i])):
			if(j==(len(t.trainTagList[i])-1)):
				newrecord="EndFlag"+"\t"+t.trainTagList[i][j]
			else:
				newrecord=t.trainSentenceList[i][j+1]+"\t"+t.trainTagList[i][j]
			if(newrecord not in nnsgramdict):
				nnsgramdict[newrecord]=1
			else:
				nnsgramdict[newrecord]=nnsgramdict[newrecord]+1

	for item in nnsgramdict:
		if(nnsgramdict[item]>5):
			nnsTagUnit=item.split("\t")
			rule4out.write("\t\t\t\t\t%s\t\t%s\n" % (nnsTagUnit[0],nnsTagUnit[1]))	