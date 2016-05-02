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