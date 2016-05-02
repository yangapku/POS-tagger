import numpy as np
from postagger import history

#number of rules
featuredim=13905

#the list to store rules
rulelist=[]

#read in rules recorded in txt file
def readRules():
	rulesReader=open('rules.txt','r',encoding='utf-8')
	for i in range(featuredim):
		rulelist.append(rulesReader.readline()[:-1].split('\t'))
	rulesReader.close()

#calculate representation for an input history and tag
def phi(h,tag):
	featureVector=np.zeros(featuredim)
	for i in range(featuredim):
		featureVector[i]=isconform(h,tag,i)
	return featureVector

#calculate representation for an input sentence and tag sequence
def totalphi(sentence,tagseq):
	globalFeatureVector=np.zeros(featuredim)
	expandSent=['BeginFlag','BeginFlag']+sentence+['EndFlag','EndFlag']
	for i in range(len(sentence)):
		globalFeatureVector+=phi(history.History(
											lastLastWord=expandSent[i],
											lastWord=expandSent[i+1],
											thisWord=expandSent[i+2],
											nextWord=expandSent[i+3],
											nextNextWord=expandSent[i+4],
											lastTag=tagseq[i-1] if i>0 else "START",
											lastLastTag=tagseq[i-2]	if i>1 else "START"
										),tagseq[i])
	return globalFeatureVector

#decide whether the <history,tag> pair obey the rule
def isconform(h,tag,ruleid):
	if(tag!=rulelist[ruleid][7]):
		return 0
	if(rulelist[ruleid][0]!="" and h.thisWord!=rulelist[ruleid][0]):
		return 0
	if(rulelist[ruleid][1]!="" and h.lastTag!=rulelist[ruleid][1]):
		return 0
	if(rulelist[ruleid][2]!="" and h.lastLastTag!=rulelist[ruleid][2]):
		return 0
	if(rulelist[ruleid][3]!="" and h.lastWord!=rulelist[ruleid][3]):
		return 0
	if(rulelist[ruleid][5]!="" and h.nextWord!=rulelist[ruleid][5]):
		return 0
	return 1