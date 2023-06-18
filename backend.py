from flask import Flask, request , jsonify

from flask import Flask, render_template
from werkzeug.utils import secure_filename
import soundfile
import os
import io
import librosa
import pandas as pd
from operator import add
import numpy as np
import pickle as pk
from scipy.stats import skew
from sklearn.decomposition import PCA
import joblib
from pydub import AudioSegment
import os
import time
from scipy.signal import butter, lfilter

from scipy.io import wavfile

lowcut = 10   #freq filter
highcut = 4000
FRAME_RATE = 16000




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'webAudioFiles/'

@app.route('/WordUpload', methods=['POST' ])
def Foodupload():
    
    # Save the audio file to disk or process it in some way
    #audio_file.save("recorded_audio.wav")
    file = request.files.get('audio')
    if file and file.filename != '': 
        dest = os.path.join(
            app.config['UPLOAD_FOLDER'], 
            secure_filename(file.filename +".wav")
        )
        # Save the file on the server.
        file.save(dest)
        FileName = request.form.get('fileName')

        #data, samplerate = soundfile.read(dest)

        

        data,samplerate = librosa.load(dest,sr=16000)
        
        soundfile.write('webAudioFiles/' + FileName, data, samplerate= 16000, subtype='PCM_32')
        data,samplerate = librosa.load('webAudioFiles/' + FileName,sr=None)

        print("sample rate = ",samplerate)

    return "success"





def get_mfccV2(name, path,SAMPLE_RATE = 16000):
    data, _ = librosa.core.load(path + name, sr = SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    try:
        ft1 = librosa.feature.mfcc(data, sr = SAMPLE_RATE, n_mfcc=30)
        ft2 = librosa.feature.zero_crossing_rate(data)[0]
        ft3 = librosa.feature.spectral_rolloff(data)[0]
        ft4 = librosa.feature.spectral_centroid(data)[0]
        ft5 = librosa.feature.spectral_contrast(data)[0]
        ft6 = librosa.feature.spectral_bandwidth(data)[0]
        ft7 = librosa.feature.spectral_flatness(data)[0]
        #S, phase = librosa.magphase(librosa.stft(data))
        #ft8 = librosa.feature.rms(S=S)[0]
        #hop_length = 512
        #oenv = librosa.onset.onset_strength(y=data, sr=SAMPLE_RATE, hop_length=hop_length)
        #ft9 = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=SAMPLE_RATE,
                                              #hop_length=hop_length)

       
        ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.median(ft1, axis = 1), np.min(ft1, axis = 1))) #180 features
        ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
        ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
        ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
        ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
        ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6))) 
        ft7_trunc = np.hstack((np.mean(ft7), np.std(ft7), skew(ft7), np.max(ft7), np.median(ft7), np.max(ft7))) #30 features. 
        #ft8_trunc = np.hstack((np.mean(ft8), np.std(ft8), skew(ft8), np.max(ft8), np.median(ft8), np.max(ft8)))
        #ft9_trunc = np.hstack((np.mean(ft9), np.std(ft9), skew(ft9), np.max(ft9), np.median(ft9), np.max(ft9)))
      #  ft7_trunc =  np.hstack((np.mean(ft7), np.std(ft7), skew(ft7), np.max(ft7), np.median(ft7), np.max(ft7)))
        return pd.Series(np.hstack((ft1_trunc ,ft2_trunc,ft3_trunc,ft4_trunc, ft5_trunc,ft6_trunc,ft7_trunc)))
    except:
        print('bad file')
        print('file = ',name)
        return pd.Series([0]*210)
    


def input_feature_food(wav_file):
  with open('models/foodParams.npy', 'rb') as f:
    means_food = np.load(f)
    stds_food = np.load(f)

  TrainPCA = pk.load(open("models/TrainPCAfood.pkl",'rb'))

  train_data = get_mfccV2("",wav_file)
  X = np.array(train_data).reshape(1, -1)
  X_scaled = (X - means_food) / stds_food
  #TrainPCA = 0
  output = TrainPCA.transform(X_scaled)
  return output


def input_feature_num(wav_file):
  with open('models/numberParams.npy', 'rb') as f:
    means_num = np.load(f)
    stds_num = np.load(f)
  TrainPCA_num = pk.load(open("models/TrainPCAnum.pkl",'rb'))

  train_data = get_mfccV2("",wav_file)
  X = np.array(train_data).reshape(1, -1)
  X_scaled = (X - means_num) / stds_num # print value and hardcode   
  #TrainPCA = 0
  output = TrainPCA_num.transform(X_scaled)
  return output

def label_int_num(output):
   label2num = {'eight': 8, 'five': 5, 'four': 4, 'nine': 9, 'one': 1, 'seven': 7, 'six': 6, 'ten': 10, 'three': 3, 'two': 2}
   y = label2num[output]
   return y

def per_unit_food(food,units):
  df = pd.read_csv('input/nutritionFeaturesExtractedFormatted.csv')
  if food == "yogurt":
    food = "Yoghurt"
  else:
    food = food.capitalize()
  nutrition_data = list(df[food])
  return [x * units for x in nutrition_data]


def get_data(food_wav,number_wav):
  
  clf_food = joblib.load("models/svm_food_model.pkl")
  clf_num = joblib.load("models/MLP_num_model.pkl")
  X_food = input_feature_food(food_wav)
  X_num = input_feature_num(number_wav)
  food = clf_food.predict(X_food)
  num = clf_num.predict(X_num)
  #print("predicted nunber = ",num[0])
  #print('Food :'+food[0],'  Quantity :'+num[0])
  num = label_int_num(num[0])
  foodItem = food[0]
  qtyItem = num
  #print("predicted nunber = ",num)
  nutrition_list = per_unit_food(food[0],num)
  return nutrition_list,foodItem,qtyItem


#add food and qty information as dictionary
def getNutrientValues(num_food = 1,food_wav = "input/filteredOutputfood/yogurt_kesavRectrain_9.wav",num_wav  = "input/filteredTestDataNumbers/five_kesavRectrain_16.wav"):
  df = pd.read_csv('input/nutritionFeaturesExtractedFormatted.csv')
  elements = df['Nutritional_groups']

  nutrition_list = [0]*30
  foodList = []
  qtyList = []
  for i in range(num_food):
    
    if(num_food == 1):
      lis,foodItem,qtyItem = get_data(food_wav,num_wav)
    else:
      lis,foodItem,qtyItem = get_data(food_wav[i],num_wav[i])
    foodList.append(foodItem)
    qtyList.append(qtyItem)
    nutrition_list = list(map(add, nutrition_list, lis))
    #print(nutrition_list)
  res = {list(elements)[i]: nutrition_list[i] for i in range(len(list(elements)))}
  return res,foodList,qtyList 



def detect_leading_silence(sound, silence_threshold=-35.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms



def trimAndExport(recordPath,outputPath):
    
    for file in os.listdir(recordPath):
        #print(file)
        sound = AudioSegment.from_file(recordPath + file, format="wav")
        start_trim = detect_leading_silence(sound)
        end_trim = detect_leading_silence(sound.reverse())

        duration = len(sound)    
        trimmed_sound = sound[start_trim:duration-end_trim]
        trimmed_sound.export(outputPath + file,format = "wav")

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bandpass_filter(buffer):
    return butter_bandpass_filter(buffer, lowcut , highcut , FRAME_RATE , order=3)



def filterTestFiles(test_record_path,test_output_path,FRAME_RATE = 16000,dataType = 'int32'):
    
    for file in os.listdir(test_output_path):
        os.remove(os.path.join(test_output_path,file))

    WAV_FILE_NAME = sorted(os.listdir(test_record_path), key=len)


    #print(WAV_FILE_NAME)

    

    for fileName in WAV_FILE_NAME:
        samplerate, data = wavfile.read(os.path.join(test_record_path, fileName))
        assert samplerate == FRAME_RATE
        filtered = np.apply_along_axis(bandpass_filter, 0, data)
        wavfile.write(os.path.join(test_output_path, fileName), samplerate, filtered.astype(dataType))



@app.route("/calculate")
def calculateNutrients():
   
    originalPath = "webAudioFiles/"
    trimmedPath =  "webAudioFilesFiltered/"
    filtered_Path = "webAudioFilesTrimmed/"

    if (os.path.isfile(originalPath+"blob.wav")): 
        os.remove(originalPath+"blob.wav")

    trimAndExport(originalPath,trimmedPath)
    filterTestFiles(trimmedPath,filtered_Path)

    foodsFile = [filtered_Path + "food.wav",filtered_Path + "food2.wav"]
    qtyFile =  [filtered_Path + "number.wav",filtered_Path + "number2.wav"]
    print(foodsFile)
    print(qtyFile)
    num_food = 2
    nurtientRes,foodList,qtyList  = getNutrientValues(num_food, foodsFile, qtyFile)
    print(foodList)
    print(qtyList)



    print("files filtered and output produced.")

    for key, value in nurtientRes.items():
      nurtientRes[key] = round(value, 2)

    
    print(nurtientRes)
    outputDict = {'food':foodList,'qty':str(qtyList)}
    outputDict = {**outputDict, **nurtientRes}

    return jsonify(outputDict)
    


        

    


@app.route("/")
def index():
    return render_template("soundRecV6.html")   


if __name__ == "__main__":
    app.run(debug=True)

