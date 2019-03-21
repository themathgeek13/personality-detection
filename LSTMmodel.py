from numpy import array
from keras.preprocessing.text import one_hot

import csv
docs=[]
with open("/home/rohan/upwork/1_PersonalityTraitsSocialMedia/personality-detection/essays.csv","rb") as csvfile:
    textread = csv.reader(csvfile, delimiter=",", quotechar='"')
    for row in textread:
        #print(",".join(row))
        docs.append(row[1])

vocab_size = 100000
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)
