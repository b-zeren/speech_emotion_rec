
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt # plots
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import noisereduce as nr


from models import *

from glob import glob


sample_audios=glob("/home/buket/Desktop/codes/kovanstaj/emotion_recognition/speech_demo/RAVDESS/*/*.wav")

four_dataset_path='/home/buket/Desktop/codes/kovanstaj/emotion_recognition/4 dataset/buket_res_model.h5'
one_dimensional_path="/home/buket/Desktop/codes/kovanstaj/emotion_recognition/1d_res/buket_best_model_1d/content/best_model_1D"
spectrogram_path="/home/buket/Desktop/codes/kovanstaj/emotion_recognition/spectrogram_model/unmapped_spec_model.pt"

first_model=four_dataset_model(four_dataset_path)
second_model=one_dimensional_model(one_dimensional_path)
third_model=spectrogram_model(spectrogram_path)



ravdess_dict={"1" : "neutral", "2" : "calm", "3" : "happy", "4" : "sad", "5" : "angry", "6" : "fear", "7" : "disgusted", "8" : "surprise"}

Y_model1=[]
Y_model2=[]
Y_model3=[]
Y_ground=[]

labels_list=list(ravdess_dict.values())

i=0

lenght=len(sample_audios)
score_first=0.0
score_second=0.0
score_third=0.0
for audio_path in sample_audios[:1]:
    print(i,"/",lenght)
    i+=1
    ground_truth=ravdess_dict[audio_path[-17]]
    first_result=first_model.predict(audio_path)["emotion"]
    second_result=second_model.predict(audio_path)["emotion"]
    third_result=third_model.predict(audio_path)["emotion"]
    print(ground_truth,first_result,second_result,third_result)
    if ground_truth==first_result:
        score_first+=1.0
    if ground_truth==second_result:
        score_second+=1.0
    if ground_truth==third_result:
        score_third+=1.0
    Y_ground.append(ground_truth)
    Y_model1.append(first_result)
    Y_model2.append(second_result)
    Y_model3.append(third_result)

score_first=score_first/lenght*100.0
score_second=score_second/lenght*100.0
score_third=score_third/lenght*100.0

print("Accuracy for first model:",score_first)
print("Accuracy for second model:",score_second)
print("Accuracy for third model:",score_third)


cm1 = confusion_matrix(y_true=Y_ground,y_pred=Y_model1, labels=labels_list)
cm2= confusion_matrix(y_true=Y_ground,y_pred=Y_model2,labels=labels_list)
cm3=confusion_matrix(y_true=Y_ground,y_pred=Y_model3,labels=labels_list)


print(cm1)
print(cm2)
print(cm3)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,
                              display_labels=labels_list)

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                              display_labels=labels_list)
disp3=ConfusionMatrixDisplay(confusion_matrix=cm3,
                              display_labels=labels_list)
disp1.plot()
plt.title("4 dataset model")
plt.show()
disp2.plot()
plt.title("1D model")
plt.show()
disp3.plot()
plt.title("Spectrogram model")
plt.show()