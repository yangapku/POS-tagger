import numpy as np
from postagger import indicator
from postagger import history

#the category of POS tags
tagCategory=[
				"AD","AS","BA","CC","CD","CS","DEC","DEG","DER","DEV","DT",
				"ETC","FW","IJ","JJ","LB","LC","M","MSP","NN","NN-SHORT",
				"NR","NR-SHORT","NT","NT-SHORT","OD","ON","P","PN","PU",
				"SB","SP","VA","VC","VE","VV"
			]
#GEN of each seen word
GENdict=dict()
GENdict['null']=['M','NN','NR','VV']
	
def solver(sentence,weightVector):
	#make lists to record max scores and correspoding ancestors
	GENsize=np.ones(len(sentence),dtype=np.int64)
	isKnown=np.zeros(len(sentence),dtype=np.bool)
	for i in range(len(sentence)):
		if(sentence[i] in GENdict):
			isKnown[i]=True
			GENsize[i]=len(GENdict[sentence[i]])
		else:
			GENsize[i]=len(GENdict['null'])
	scorelist=[np.zeros((GENsize[i],GENsize[i-1])) for i in range(1,len(sentence))]
	scorelist=[np.zeros((GENsize[0],1))]+scorelist
	ancestorlist=[np.zeros((GENsize[i],GENsize[i-1])) for i in range(1,len(sentence))]
	ancestorlist=[np.zeros((GENsize[0],1))]+ancestorlist
	#add flag words to begining and end of the sentence, "BeginFlag" and "EndFlag" are flag words
	expandSent=['BeginFlag','BeginFlag']+sentence+['EndFlag','EndFlag']
	#calculate maximum score for each state in the list
	for i in range(len(scorelist)):
		for j in range(scorelist[i].shape[0]):
			for k in range(scorelist[i].shape[1]):
				#calculate score for every possible history of the first word
				if(i==0):
					scorelist[i][j,k]=float(indicator.phi(history.History(
																lastLastWord=expandSent[i],
																lastWord=expandSent[i+1],
																thisWord=expandSent[i+2],
																nextWord=expandSent[i+3],
																nextNextWord=expandSent[i+4]
															),GENdict[sentence[i]][j] if isKnown[i] else GENdict['null'][j])
															.dot(weightVector[:,np.newaxis]))
				#calculate maximum score and ancestor for every possible history of other words using DP
				else:
					tempPhis=np.zeros(scorelist[i-1].shape[1])
					for u in range(tempPhis.size):
						if(i==1):
							tempPhis[u]=float(indicator.phi(history.History(
																lastLastWord=expandSent[i],
																lastWord=expandSent[i+1],
																thisWord=expandSent[i+2],
																nextWord=expandSent[i+3],
																nextNextWord=expandSent[i+4],
																lastTag=GENdict[sentence[i-1]][k] if isKnown[i-1] else GENdict['null'][k]
															),GENdict[sentence[i]][j] if isKnown[i] else GENdict['null'][j])
															.dot(weightVector[:,np.newaxis]))
						else:
							tempPhis[u]=float(indicator.phi(history.History(
																lastLastWord=expandSent[i],
																lastWord=expandSent[i+1],
																thisWord=expandSent[i+2],
																nextWord=expandSent[i+3],
																nextNextWord=expandSent[i+4],
																lastTag=GENdict[sentence[i-1]][k] if isKnown[i-1] else GENdict['null'][k],
																lastLastTag=GENdict[sentence[i-2]][u] if isKnown[i-2] else GENdict['null'][u]
															),GENdict[sentence[i]][j] if isKnown[i] else GENdict['null'][j])
															.dot(weightVector[:,np.newaxis]))
					scorelist[i][j,k]=np.max(tempPhis+scorelist[i-1][k])
					ancestorlist[i][j,k]=np.argmax(tempPhis+scorelist[i-1][k])
	#fill in output tag sequence using the two lists
	outputTagId=[-1 for i in range(len(scorelist))]
	outputTagId[-1]=int(scorelist[-1].argmax()//scorelist[-1].shape[1])
	if(len(outputTagId)>1):
		outputTagId[-2]=int(scorelist[-1].argmax()%scorelist[-1].shape[1])
		for i in range(len(outputTagId)-2):
			outputTagId[i]=int(ancestorlist[i+2][outputTagId[i+2],outputTagId[i+1]])
	outputTags=[GENdict[sentence[i]][outputTagId[i]] if sentence[i] in GENdict else GENdict['null'][outputTagId[i]] 
					for i in range(len(sentence))]
	return outputTags
	
	
	
	
	
	
	
	
	
	
	
	
	
