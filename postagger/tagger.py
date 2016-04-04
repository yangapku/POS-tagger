import numpy as np
from postagger import indicator
from postagger import decoder

class ChineseTagger:
	def __init__(self):
		#final average alpha vector for perceptron
		self.learnedWeightVector=np.zeros(indicator.featuredim)
		self.learnRate=1.0
		#lists to store training set data
		self.trainSentenceList=[]
		self.trainTagList=[]
		self.perceptronMode='Avg'
	
	def trainTagger(self,sentencepath='dataset/trn.wrd',tagpath='dataset/trn.pos',learnRate=1.0,perceptronMode='Avg',maxIteration=2,eps=0.001):
		#decide using final or average weight as output for predicting
		if(not perceptronMode in {'Avg','Final'}):
			print('error:Illegal perceptronMode setting.')
			return;
		self.perceptronMode=perceptronMode
		#set learn rate
		self.learnRate=learnRate
		#temp vectors to store intermediate results
		sumWeightVector=np.zeros(indicator.featuredim)
		tempWeightVector=np.zeros(indicator.featuredim)
		#readin training set data
		try:
			#open training text files
			trnwrdopen=open(sentencepath,'r',encoding='utf-8')
			trnposopen=open(tagpath,'r',encoding='utf-8')
			tempWordSeq=[]
			tempTagSeq=[]
			wrdReadLine='spam'
			posReadLine='spam'
			while(wrdReadLine!=""):
				#readin a line
				wrdReadLine=trnwrdopen.readline()
				posReadLine=trnposopen.readline()
				#process '\n'
				if(wrdReadLine!=''):
					word=wrdReadLine[:-1]
				else:
					word=wrdReadLine
				if(posReadLine!=''):
					pos=posReadLine[:-1]
				else:
					pos=posReadLine
				#append word into sentence or add sentence into sentence list
				if(word!=''):
					tempWordSeq.append(word)
					tempTagSeq.append(pos)
				else:
					if(len(tempWordSeq)!=0):
						self.trainSentenceList.append(tempWordSeq)
						self.trainTagList.append(tempTagSeq)
						#begin a new sentence
						tempWordSeq=[]
						tempTagSeq=[]
			print('Successfully read in '+str(len(self.trainSentenceList))+' training sentences.')
			#close training files
			trnwrdopen.close()
			trnposopen.close()
		except Exception as e:
			print('error:Failed to load training set.')
		#do iteration to optimize weight vector
		if(perceptronMode=='Avg'):
			for i in range(maxIteration):
				sumWeightVector=np.zeros(indicator.featuredim)
				for j in range(len(self.trainSentenceList)):
					#estimate tag sequence using Viterbi
					estimatedTagSeq=decoder.solver(self.trainSentenceList[j])
					#update tempWeightVector
					if(estimatedTagSeq!=self.trainTagList[j]):
						tempWeightVector=tempWeightVector+indicator.totalphi(self.trainSentenceList[j],self.trainTagList[j])-indicator.totalphi(self.trainSentenceList[j],estimatedTagSeq)
					sumWeightVector+=tempWeightVector
				resisualSqsum=np.sum((self.learnedWeightVector-sumWeightVector/len(self.trainSentenceList))**2)
				if(resisualSqsum<eps):
					break
		'''To do:implement Final mode'''
		print('Successfully trained weight vector.')
		
	def developTagger(self):
		pass
	
	def clearTrainRecord(self):
		self.learnedWeightVector=np.zeros(indicator.featuredim)
		self.learnRate=1.0
		self.trainSentenceList=[]
		self.trainTagList=[]
		self.perceptronMode='Avg'
		print('The training record is fully removed.')
		
	def predictTag(self,sentenceList):
		return(1)