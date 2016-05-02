from postagger import history
from postagger import tagger
from postagger import decoder
from postagger import indicator
from postagger import history
import numpy as np

t=tagger.ChineseTagger()
t.readinTrainData()

with open('rule_lastwordS.txt','w',encoding="utf-8") as rule5out:
	lsgramdict=dict()
	for i in range(len(t.trainTagList)):
		for j in range(len(t.trainTagList[i])):
			thisword=t.trainSentenceList[i][j]
			if(len(decoder.GENdict[thisword])>1):
				if(j==0):
					newrecord=thisword+"\t"+"BeginFlag"+"\t"+t.trainTagList[i][0]
				else:
					newrecord=thisword+"\t"+t.trainSentenceList[i][j-1]+"\t"+t.trainTagList[i][j]
				if(newrecord not in lsgramdict):
					lsgramdict[newrecord]=1
				else:
					lsgramdict[newrecord]=lsgramdict[newrecord]+1

	for item in lsgramdict:
		if(lsgramdict[item]>5):
			lsTagUnit=item.split("\t")
			rule5out.write("%s\t\t\t%s\t\t\t\t%s\n" % (lsTagUnit[0],lsTagUnit[1],lsTagUnit[2]))