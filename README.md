# recognizingFace
import cv2
import glob
import random
import numpy as np
emotions = ["normal", "angry", "contempt", "disgust", "fear", "happy", "sad", "surprise"] 
fishface = cv2.createFisherFaceRecognizer()
data = {}
def getfiles(emotion): 
    files = glob.glob("dataset\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] 
    prediction = files[-int(len(files)*0.2):] 
    return training, prediction
def makesets():
    trainingdata = []
    traininglabels = []
    predictiondata = []
    predictionlabels = []
    for emotion in emotions:
        training, prediction = getfiles(emotion)
        
        for item in training:
            image = cv2.imread(item) 
            gray = cv2.cvtColor(image, cv2.COLORBGR2GRAY) 
            trainingdata.append(gray) 
            traininglabels.append(emotions.index(emotion))
        for item in prediction: 
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLORBGR2GRAY)
            predictiondata.append(gray)
            predictionlabels.append(emotions.index(emotion))
    return trainingdata, traininglabels, predictiondata, predictionlabels
def runrecognizer():
    trainingdata, traininglabels, predictiondata, predictionlabels = makesets()
    print "training fisher face classifier"
    print "size of training set is:", len(traininglabels), "images"
    fishface.train(trainingdata, np.asarray(traininglabels))
    print "predicting classification set"
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == predictionlabels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100*correct)/(correct + incorrect))
metascore = []
for i in range(0,10):
    correct = runrecognizer()
    print "got", correct, "perfect!"
    metascore.append(correct)
print "\n\nover:", np.mean(metascore), "pefect"

