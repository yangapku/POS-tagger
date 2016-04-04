import numpy as np
from postagger import indicator

#the category of POS tags
tagCategory=[
				"AD","AS","BA","CC","CD","CS","DEC","DEG","DER","DEV","DT",
				"ETC","FW","IJ","JJ","LB","LC","M","MSP","NN","NN-SHORT",
				"NR","NR-SHORT","NT","NT-SHORT","OD","ON","P","PN","PU",
				"SB","SP","VA","VC","VE","VV"
			]

#a class used to store history info, "START" is a flag tag
class History(object):
	def __init__(
					self,
					lastLastWord,
					lastWord,
					thisWord,
					nextWord,
					nextNextWord,
					lastTag="START",
					lastLastTag="START"					
				):
		self.lastLastTag=lastLastTag
		self.lastTag=lastTag
		self.lastLastWord=lastLastWord
		self.lastWord=lastWord
		self.thisWord=thisWord
		self.nextWord=nextWord
		self.nextNextWord=nextNextWord
	
def solver(sentence,weightVector):
	#make lists to record max scores and correspoding ancestors
	scorelist=[np.zeros((len(tagCategory),len(tagCategory))) for i in range(len(sentence))]
	scorelist[0]=np.zeros((len(tagCategory),1))
	ancestorlist=[np.zeros((len(tagCategory),len(tagCategory))) for i in range(len(sentence))]
	ancestorlist[0]=np.zeros((len(tagCategory),1))
	#add flag words to begining and end of the sentence, "BeginFlag" and "EndFlag" are flag words
	expandSent=['BeginFlag','BeginFlag']+sentence+['EndFlag','EndFlag']
	#calculate maximum score for each state in the list
	for i in range(len(scorelist)):
		for j in range(scorelist[i].shape[0]):
			for k in range(scorelist[i].shape[1]):
				#calculate score for every possible history of the first word
				if(i==0):
					
					scorelist[i][j,k]=float(indicator.phi(History(
																lastLastWord=expandSent[i],
																lastWord=expandSent[i+1],
																thisWord=expandSent[i+2],
																nextWord=expandSent[i+3],
																nextNextWord=expandSent[i+4]
															),tagCategory[j]).dot(weightVector[:,np.newaxis]))
				#calculate maximum score and ancestor for every possible history of other words using DP
				else:
					tempPhis=np.zeros(scorelist[i-1].shape[1])
					for u in range(tempPhis.size):
						if(i==1):
							tempPhis[u]=float(indicator.phi(History(
																lastLastWord=expandSent[i],
																lastWord=expandSent[i+1],
																thisWord=expandSent[i+2],
																nextWord=expandSent[i+3],
																nextNextWord=expandSent[i+4],
																lastTag=tagCategory[k]
															),tagCategory[j]).dot(weightVector[:,np.newaxis]))
						else:
							tempPhis[u]=float(indicator.phi(History(
																lastLastWord=expandSent[i],
																lastWord=expandSent[i+1],
																thisWord=expandSent[i+2],
																nextWord=expandSent[i+3],
																nextNextWord=expandSent[i+4],
																lastTag=tagCategory[k],
																lastLastTag=tagCategory[u]
															),tagCategory[j]).dot(weightVector[:,np.newaxis]))
					scorelist[i][j,k]=np.max(tempPhis+scorelist[i-1][k])
					ancestorlist[i][j,k]=np.argmax(tempPhis+scorelist[i-1][k])
	#fill in output tag sequence using the two lists
	outputTagId=[-1 for i in range(len(scorelist))]
	outputTagId[-1]=int(scorelist[-1].argmax()//scorelist[-1].shape[1])
	if(len(outputTagId)>1):
		outputTagId[-2]=int(scorelist[-1].argmax()%scorelist[-1].shape[1])
		for i in range(len(outputTagId)-2):
			outputTagId[i]=int(ancestorlist[i+2][outputTagId[i+2],outputTagId[i+1]])
	outputTags=[tagCategory[i] for i in outputTagId]
	return outputTags
	
	
	
	
	
	
	
	
	
	
	
	
	
