from postagger import history
from postagger import tagger
from postagger import decoder
from postagger import indicator
from postagger import history
import numpy as np

t=tagger.ChineseTagger()
t.readinTrainData()

with open('rule_lastwordNS.txt','w',encoding="utf-8") as rule3out:
	lnsgramdict=dict()
	for i in range(len(t.trainTagList)):
		for j in range(len(t.trainTagList[i])):
			if(j==0):
				newrecord="BeginFlag"+"\t"+t.trainTagList[i][0]
			else:
				newrecord=t.trainSentenceList[i][j-1]+"\t"+t.trainTagList[i][j]
			if(newrecord not in lnsgramdict):
				lnsgramdict[newrecord]=1
			else:
				lnsgramdict[newrecord]=lnsgramdict[newrecord]+1

	for item in lnsgramdict:
		if(lnsgramdict[item]>5):
			lnsTagUnit=item.split("\t")
			rule3out.write("\t\t\t%s\t\t\t\t%s\n" % (lnsTagUnit[0],lnsTagUnit[1]))	