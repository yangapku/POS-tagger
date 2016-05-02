from postagger import history
from postagger import tagger
from postagger import decoder
from postagger import indicator
from postagger import history
import numpy as np

t=tagger.ChineseTagger()
t.readinTrainData()

with open('rule_lasttagS.txt','w',encoding="utf-8") as rule6out:
	ltsgramdict=dict()
	for i in range(len(t.trainTagList)):
		for j in range(len(t.trainTagList[i])):
			thisword=t.trainSentenceList[i][j]
			if(len(decoder.GENdict[thisword])>1):
				if(j==0):
					newrecord=thisword+"\t"+"START"+"\t"+t.trainTagList[i][0]
				else:
					newrecord=thisword+"\t"+t.trainTagList[i][j-1]+"\t"+t.trainTagList[i][j]
				if(newrecord not in ltsgramdict):
					ltsgramdict[newrecord]=1
				else:
					ltsgramdict[newrecord]=ltsgramdict[newrecord]+1

	for item in ltsgramdict:
		if(ltsgramdict[item]>5):
			ltsTagUnit=item.split("\t")
			rule6out.write("%s\t%s\t\t\t\t\t\t%s\n" % (ltsTagUnit[0],ltsTagUnit[1],ltsTagUnit[2]))