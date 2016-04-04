from postagger import tagger

#build a tagger object
t=tagger.ChineseTagger()

#train and develop the tagger
t.trainTagger()
t.developTagger()

#predict tag for input sentence
result=t.predictTag(['欢迎','来','美丽','的','中国','玩','。'])
print(result)