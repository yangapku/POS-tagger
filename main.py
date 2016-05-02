from postagger import tagger

#build a tagger object
t=tagger.ChineseTagger()

#train and develop the tagger
t.readinTrainData()
t.trainTagger()

#predict tag for input sentence
t.developTagger()
t.predictTag()