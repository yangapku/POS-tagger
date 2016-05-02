import numpy as np
from postagger import indicator
from postagger import decoder

class ChineseTagger(object):
	def __init__(self):
		#final average alpha vector for perceptron
		self.learnedWeightVector=np.zeros(indicator.featuredim)
		#readin existing training parameters
		with open('parameters.txt','r',encoding="utf-8") as parain:
			line=parain.readline()
			if(line[0]=="Y"):
				for i in range(indicator.featuredim):
					line=parain.readline()
					self.learnedWeightVector[i]=float(line[:-1])
				print("Successfully readin existing parameters.")
		self.learnRate=1.0
		#lists to store training set data
		self.trainSentenceList=[]
		self.trainTagList=[]
		self.perceptronMode='Avg'
		#initialize rulelist
		indicator.readRules()
	
	def readinTrainData(self,sentencepath='dataset/trn.wrd',tagpath='dataset/trn.pos'):
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
						#add word-pos pair into GEN
						for i in range(len(tempWordSeq)):
							if(tempWordSeq[i] not in decoder.GENdict):
								decoder.GENdict[tempWordSeq[i]]=set([tempTagSeq[i]])
							else:
								decoder.GENdict[tempWordSeq[i]].add(tempTagSeq[i])
						#begin a new sentence
						tempWordSeq=[]
						tempTagSeq=[]
			for item in decoder.GENdict:
				decoder.GENdict[item]=list(decoder.GENdict[item])
			print('Successfully read in '+str(len(self.trainSentenceList))+' training sentences.')
			#close training files
			trnwrdopen.close()
			trnposopen.close()
		except Exception as e:
			print('error:Failed to load training set.')
	
	def trainTagger(self,learnRate=1.0,perceptronMode='Avg',maxIteration=7,eps=1):
		#decide using final or average weight as output for predicting
		if(not perceptronMode in {'Avg','Final'}):
			print('error:Illegal perceptronMode setting.')
			return;
		self.perceptronMode=perceptronMode
		#set learn rate
		self.learnRate=learnRate
		#do iteration to optimize weight vector
		sumWeightVector=np.zeros(indicator.featuredim)
		tempWeightVector=self.learnedWeightVector
		for i in range(maxIteration):
			for j in range(len(self.trainSentenceList)):
				#estimate tag sequence using Viterbi
				estimatedTagSeq=decoder.solver(self.trainSentenceList[j],tempWeightVector)
				#update tempWeightVector
				if(estimatedTagSeq!=self.trainTagList[j]):
					tempWeightVector=tempWeightVector+learnRate*(indicator.totalphi(self.trainSentenceList[j],self.trainTagList[j])-indicator.totalphi(self.trainSentenceList[j],estimatedTagSeq))
				sumWeightVector+=tempWeightVector
			vectorAfterIteration=sumWeightVector/((i+1)*len(self.trainSentenceList))
			resisualSqsum=np.sum((self.learnedWeightVector-vectorAfterIteration)**2)
			self.learnedWeightVector=vectorAfterIteration
			if(resisualSqsum<eps):
				break
		#write trained parameters
		with open('parameters.txt','w',encoding="utf-8") as paraout:
			paraout.write('Y\n')
			for i in range(indicator.featuredim):
				paraout.write(str(self.learnedWeightVector[i])+"\n")
		print('Successfully trained weight vector.')
		
	def developTagger(self,sentencePath="dataset/dev.wrd",outPath="dataset/devpred.pos"):
		self.predictTag(sentencePath,outPath,'develop')
	
	def clearTrainRecord(self):
		self.learnedWeightVector=np.zeros(indicator.featuredim)
		self.learnRate=1.0
		self.trainSentenceList=[]
		self.trainTagList=[]
		self.perceptronMode='Avg'
		print('The training record is fully removed.')
		
	def predictTag(self,sentencePath="dataset/tst.wrd",outPath="dataset/tst.pos",usage="test"):
		testSentenceList=[]
		testin=open(sentencePath,'r',encoding="utf-8")
		testout=open(outPath,'w',encoding="utf-8")
		tempWordSeq=[]
		wrdReadLine='spam'
		while(wrdReadLine!=""):
			wrdReadLine=testin.readline()
			if(wrdReadLine!=''):
				word=wrdReadLine[:-1]
			else:
				word=wrdReadLine
			if(word!=''):
				tempWordSeq.append(word)
			else:
				if(len(tempWordSeq)!=0):
					testSentenceList.append(tempWordSeq)
					#begin a new sentence
					tempWordSeq=[]
		print('Successfully read in '+str(len(testSentenceList))+' '+usage+' sentences.')
		for item in testSentenceList:
			result=decoder.solver(item,self.learnedWeightVector)
			for eachpos in result:
				testout.write(eachpos+"\n")
			testout.write("\n")
			count=count+1
		testin.close()
		testout.close()
		print('Successfully write in output result file.')