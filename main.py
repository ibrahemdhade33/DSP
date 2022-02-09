import matplotlib.pyplot as plt
import scipy.signal
from scipy import signal
from scipy.signal import filtfilt
from scipy.io import wavfile
import numpy as np
import xlrd
import pyaudio
from scipy.fft import fft
from scipy.signal import find_peaks

def playaudio(sample_rate, samples):
   frequencies, times, spectrogram = signal.spectrogram(samples,sample_rate)
   p = pyaudio.PyAudio()
   stream = p.open(format=pyaudio.paFloat32,
                   channels=1,
                   rate=sample_rate,
                   output=True)

   stream.write(samples)
   stream.stop_stream()
   stream.close()
   p.terminate()
def readauido():
   sample_rate, samples = wavfile.read('C:/Users/Ibrah/PycharmProjects/dsp_python_project/string_5.wav')
   return sample_rate,samples
def readfile():
   dec={}
   loc = ('char.xlsx')
   wb = xlrd.open_workbook(loc)
   sheet = wb.sheet_by_index(0)

   sheet.cell_value(0, 0)
   for i in range(sheet.nrows):
      s = sheet.row_values(i)[0]
      s1 = sheet.row_values(i)[1]
      l = s.split("/")
      l1 = list(map(int, s1.split("/")))
      dec[l[0]] = list(map(int,[l1[0] , sheet.row_values(i)[2], sheet.row_values(i)[3], sheet.row_values(i)[4]]))
      dec[l[1]] = list(map(int,[l1[1] , sheet.row_values(i)[2], sheet.row_values(i)[3], sheet.row_values(i)[4]]))

   return dec
dec=readfile()
def getfrequancies():
   s = set()
   for i in dec:
      for j in dec[i] :
         s.add(j)
   s = list(s)
   s = sorted(s)
   s = s[1:]
   return s
def bandpassfilter():
   s = getfrequancies()
   fs=8000
   order=2
   list_of_filter=[]
   for i in s :
      m=i
      low = m-10
      hieght = m+10

      lowc =(2*low)/fs
      hieghtc=(2*hieght)/fs
      if hieght >=4000 :
         [b,a] = scipy.signal.butter(order,lowc,btype='hp')
      else:
         [b,a] = scipy.signal.butter(order,[lowc,hieghtc],btype='bandpass')
      [w,h]=scipy.signal.freqz(b,a)
      w=fs*w/(2*np.pi)

      plt.plot(w,abs(h))
      plt.axis([m-100,m+100,-2,2])
      plt.xlabel("Frequany(Hz)")
      plt.ylabel("Magnitude")
      if i ==4000 :
         plt.title(f"Height pass filter for {i}Hz")
      else :
         plt.title(f"Band pass filter for {i}Hz")
      plt.show()
      list_of_filter.append([b,a])



   return list_of_filter



def encode():

   s=input("please enter a string to encode it \n")
   s=s.strip("\n")
   print("to decode the string please chose the second choice")
   Fs = 8000
   f=Fs*0.04
   t = np.arange(0,f)
   encode_string =[]
   t1 = np.arange(0,f*len(s),1)

   for i in range(0,len(s)) :
      encode_string.append(np.cos(2*np.pi*dec[s[i]][0]*t/Fs)+np.cos(2*np.pi*dec[s[i]][1]*t/Fs)+np.cos(
         2*np.pi*dec[s[i]][2]*t/Fs) + np.cos(2*np.pi*dec[s[i]][3]*t/Fs))
   allstring =[]
   for i in encode_string :
      for j in i :
         allstring.append(j)

   plt.plot(t1,allstring)
   plt.xlabel("Sample")
   plt.ylabel("Magnitude")
   plt.title("Encoded String Cos wave")
   plt.show()
   arr=np.array(allstring)
   wavfile.write('cos.wav',Fs,arr)
   sample_rate, samples = readauido()
   playaudio(sample_rate,samples)



def decodefilter():

   s=getfrequancies()
   sample_rate, samples = readauido()
   playaudio(sample_rate, samples)
   list_of_filter =bandpassfilter()

   period = int(sample_rate * 0.04)
   i = 0
   j = period
   decode_string = ""

   for k in range(0, len(samples)//period ):
      l = []
      b = []
      for m in range(i, j):
         l.append(samples[m])
      arr = np.array(l)

      for n in list_of_filter:
         y1 = scipy.signal.lfilter(n[0], n[1], arr, axis=0)
         y1 =np.sum(y1*y1)

         b.append(y1)

      maximums=[]

      x=sorted(b)
      m=x[-1]
      m1=x[-2]
      m3=x[-3]
      m4=x[-4]
      maximums.append(s[b.index(m)])
      maximums.append(s[b.index(m1)])
      maximums.append(s[b.index(m3)])
      maximums.append(s[b.index(m4)])

      i += period
      j += period
      if len(maximums)==3:
         maximums.append(0)
      maximums=sorted(maximums)
      for o in dec :
         l=sorted(dec[o])
         if l[0]==maximums[0] and l[1]==maximums[1] and l[2] == maximums[2] and l[3] == maximums[3]:
            decode_string+=o
   print(decode_string)








def decodefft():

   sample_rate, samples = readauido()
   print(len(samples) , sample_rate)
   playaudio(sample_rate,samples)
   plt.plot(samples)
   plt.show()
   period=int(sample_rate*0.04)
   i=0
   j=320
   decode_string =""
   c=0
   for k in range(0,len(samples)//period):
      c+=1
      l=[]
      for m in range(i,j):
         l.append(samples[m])
      arr=np.asarray(l)
      y=fft(arr)
      y=np.abs(y)

      y=y[0:((period)//2 +10)]
      i+=period
      j+=period
      peaks,_=find_peaks(y,height=1,threshold=4)

      for p  in range(0,len(peaks)):
         peaks[p]*=25
      print(peaks)
      peaks=list(map(int,peaks))

      print(c)
      if len(peaks)==3:
         peaks.append(0)
      peaks =sorted(peaks)

      for o in dec :
         l=sorted(dec[o])
         if l[0] == peaks[0] and l[1] == peaks[1] and l[2] == peaks[2] and l[3] == peaks[3]:
            decode_string+=o

   plt.plot(np.abs(fft(samples)))
   plt.xlabel("Frequency")
   plt.ylabel("Magnitude")
   plt.axis()
   plt.title("Fourier Transform for all wav signal ")


   plt.show()
   print(decode_string)

while True:
   c=input("\tplease chose from the menu\n--------------------------------------\n"
         "1-Encode String\n"
         "2-Decode String\n"
         "3-exit\n--------------------------------------\n")
   if c== "1":
      encode()
   elif c=="2":
      c= input("please chose\n"
            "1-fft decoding\n"
            "2-filter decoding\n")
      if c== "1" :
         decodefft()
      elif c=="2":
         decodefilter()
      else:
         print("error choice !!")

   elif c=="3":
      print("bye bye :):)\n")
      exit(0)
   else:
      print("error choice please chose again")