import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt # plots
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
import seaborn as sns
import noisereduce as nr

from glob import glob
import os

from models import *


four_dataset_path='/home/buket/Desktop/codes/kovanstaj/4 dataset/buket_res_model.h5'
one_dimensional_path="/home/buket/Desktop/codes/kovanstaj/1d_res/buket_best_model_1d/content/best_model_1D"

first_model=four_dataset_model(four_dataset_path)
second_model=one_dimensional_model(one_dimensional_path)


iemocap_path="/home/buket/Desktop/codes/kovanstaj/emotion_recognition/speech_demo/IEMOCAP/IEMOCAP_full_release"


labels=pd.read_csv("/home/buket/Desktop/codes/kovanstaj/emotion_recognition/speech_demo/IEMOCAP/iemocap_full_dataset.csv")
labels=labels.drop(columns=['session','method','gender','n_annotators','agreement'])


labels=labels.replace({'neu':'neutral', 'fru':"angry", 'sad':"sad", 'sur':'surprise', 'ang':"angry", 'hap':'happy', 'exc':"happy", 'fea':"fear", 'dis':"disgusted",'oth':"other"})


labels_list=labels["emotion"].unique()

Y_ground=[]
Y_first=[]
Y_second=[]
score_first=0.0
score_second=0.0
lengh=len(labels)
for idx,row in labels[:100].iterrows():
    audio_path=os.path.join(iemocap_path,row["path"])
    
    ground_truth=row["emotion"]
    first_result=first_model.predict(audio_path)
    second_result=second_model.predict(audio_path)

    print(idx,"/",lengh)
    if ground_truth==first_result["emotion"]:
        score_first+=1.0
    if ground_truth==second_result["emotion"]:
        score_second+=1.0
    Y_ground.append(ground_truth)
    Y_first.append(first_result["emotion"])
    Y_second.append(second_result["emotion"])

score_first=score_first/lengh*100.0
score_second=score_second/lengh*100.0 

print("Accuracy for first model:",score_first)
print("Accuracy for second model:",score_second)


cm1 = confusion_matrix(y_true=Y_ground,y_pred=Y_first, labels=labels_list)
cm2= confusion_matrix(y_true=Y_ground,y_pred=Y_second,labels=labels_list)


print(cm1,cm2)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,
                              display_labels=labels_list)

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                              display_labels=labels_list)
disp1.plot()
plt.show()
disp2.plot()
plt.show()

