import logging
import os
import sys

import numpy as np

USE_L2_METRIC = False

def get_roc(predict_file,roc_file):
    print("...start to calculate for roc")
    posi_simi_metric = []
    nega_simi_metric = []
    with open(predict_file,"r") as f:
         predicts = f.readlines()
         for line in predicts:
             line = line.strip()
             if(len(line) > 0):
               
               arr = line.split(',')
               print(arr)
               ground_label = arr[0]
               simi_metric = arr[1]
               if((int)(ground_label) == 1):
                  posi_simi_metric.append(simi_metric)
               else:
                  nega_simi_metric.append(simi_metric)
    
    print 'posi_size = '+ str(len(posi_simi_metric))+' nega_size = '+str(len(nega_simi_metric))
     
    choosed_th=0
    if USE_L2_METRIC:
         a=1
    else:
        to_print = 0
        th = np.arange(0.1,0.4,0.0001)
        with open(roc_file,'w') as f:
            for i in th:
                TP = 0
                FP = 0
                for m in posi_simi_metric:
                    if float(m)>float(i):
                        TP = TP+1
                for n in nega_simi_metric:
                    if float(n)>float(i):
                        FP = FP+1
                f.write(str(i)+' '+str((TP)/(0.000001+len(posi_simi_metric)))+' '+str((FP)/(0.000001+len(nega_simi_metric)))+'\n')   
                if FP>=TP and to_print==0:
                    print 'EER is '+str((TP+FP)/5474.0)+' and the th is '+ str(i)
                    to_print = 1
                    choosed_th = i
        

def main():
    predict_file = "rs.txt"
    roc_file = "roc.txt"
    veri_err = "veri_err.txt"
    get_roc(predict_file,roc_file)

if __name__ == '__main__':
    main()
