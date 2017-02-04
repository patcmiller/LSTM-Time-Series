import pandas as pd
import numpy as np
import re, math, sys, time, os, lstm
import gen_outliers as gen_outliers
import matplotlib.pyplot as plt
import glob
plt.ioff()
        
def getFileNames(dir_csvs, fileType, dataFrom=False):
  extension= fileType    
  path= dir_csvs
  
  allFiles= []
  for root, dirs_list, files_list in os.walk(path):
    for file_name in files_list:
      if os.path.splitext(file_name)[-1] == extension:
        file_name_path = os.path.join(root, file_name)
        allFiles.append(file_name_path)
  
  myA1Files= [x for x in allFiles if ('all' not in x and 'A1' in x)]
  myA2Files= [x for x in allFiles if ('all' not in x and 'A2' in x)]
  myA3Files= [x for x in allFiles if ('all' not in x and 'A3' in x)]
  myA4Files= [x for x in allFiles if ('all' not in x and 'A4' in x)]
  return myA1Files, myA2Files, myA3Files, myA4Files

def plotAnomalies(df, title, p, to_plot, use_prediction, print_detection=False):
  df['outliers']= gen_outliers.getOutliers(df[to_plot])
  if print_detection: print(title, gen_outliers.compareAnomalies(df.is_anomaly.values, df.outliers.values))
  
  dot_size= 20.0
  p.plot(df.timestamp, df[to_plot], color='k', zorder=1)
  if use_prediction: p.plot(df.timestamp, df.prediction, color='orange', zorder=2)
  p.scatter(df.timestamp.ix[(df.outliers==1) & (df.is_anomaly==0)], \
    df[to_plot].ix[(df.outliers==1) & (df.is_anomaly==0)], marker='8', s=dot_size, color='cyan', zorder=3) # ANOMALIES PREDICTED not REAL
  p.scatter(df.timestamp.ix[(df.outliers==0) & (df.is_anomaly==1)], \
    df[to_plot].ix[(df.outliers==0) & (df.is_anomaly==1)], marker='8', s=dot_size, color='red', zorder=4) # ANOMALIES REAL not DETECTED
  p.scatter(df.timestamp.ix[(df.outliers==1) & (df.is_anomaly==1)], \
    df[to_plot].ix[(df.outliers==1) & (df.is_anomaly==1)], marker='8', s=dot_size, color='lime', zorder=5) # ANOMALIES DETECTED
  p.set_title(title)

def graphFiles(names, tag, n_timestamp, n_value, n_anomaly):
  numFiles= len(names)
  rows, cols= 1, 2
  
  p,plots= plt.subplots(nrows=2*rows,ncols=cols,figsize=(15,8))
  # wm= plt.get_current_fig_manager()
  # wm.window.wm_geometry("1500x834+0+10")
  counter= 0

  for n in names:
    title= re.sub(r'\W+', '', n)[:-3]
    print(title)
    
    df= pd.read_csv(n,'r',delimiter=',')
    df['timestamp']= df[n_timestamp]
    if df.timestamp.dtype=='object': df.timestamp= range(df.shape[0])

    df['value']= df[n_value]
    df['is_anomaly']= df[n_anomaly]
    
    preds= lstm.runLstm(n)
    df['prediction']= preds.flatten()
    df['error']= np.subtract(df.value.values,df.prediction.values)**2
    
    col= counter%cols
    row1= int(counter/cols)
    row2= int((counter+2.0)/cols)
    plotAnomalies(df, title, plots[row1,col], 'value', True, False)
    plotAnomalies(df, title, plots[row2,col], 'error', False, False)       

    counter+=1
    if counter== rows*cols or n==names[-1]:
      plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
      plt.savefig('graphs/'+tag+'/'+tag+'_'+title+'.png')
      plt.close()
      
      p,plots= plt.subplots(nrows=2*rows,ncols=cols,figsize=(15,8))
      # wm= plt.get_current_fig_manager()
      # wm.window.wm_geometry("1500x834+0+10")
      counter= 0

def main():
  d1,d2,d3,d4= getFileNames('ydata','.csv','Yahoo')
  graphFiles(d1,'A1','timestamp','value','is_anomaly')
  graphFiles(d2,'A2','timestamp','value','is_anomaly')
  graphFiles(d3,'A3','timestamps','value','anomaly')
  graphFiles(d4,'A4','timestamps','value','anomaly')
    
main()