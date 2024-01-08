# -*- coding: utf-8 -*-
"""SimBot

   Simulation agent for montecarlo simulations
"""


"""# packages Inicialization """

from numba import cuda
import cudf as cdf
import cupy as cpy
import pandas as pd
import numpy as np
from math import sqrt
import locale
from datetime import datetime
import os
import glob
import threading

locale.getpreferredencoding=lambda:"UTF-8"

#Global var - shared memory for threads
simItems = pd.DataFrame()

"""# Calculation Routines

## GPU Routines
"""

# GPU Routines
def save_statsGPU(stats, texto):
  dt = datetime.now()
  st = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
  stats=cdf.concat([stats, cdf.Series({"DataHora": st, "Msg":texto})], ignore_index=True)
  return stats

def calculate_statsGPU(sim, n):
      mean = sim["PTotal"].mean()

      squareErrorSum = 0

      ers = ((sim["PTotal"]-mean)**2)
      squareErrorSum = (ers.sum())

      if (n>1):
          stdDev = sqrt(squareErrorSum/(n-1))
      else:
          stdDev = 1

      zc = 1.96
      error = (zc*stdDev)/sqrt(n)
      interv_Inferior = mean - error
      interv_Superior = mean + error
      tolerance = abs(error / mean)

      st = [mean, stdDev, zc, interv_Inferior, interv_Superior, error, tolerance]

      return st

def createRandomGPU(nFields, upperLimit, size):
  cpy.random.seed(None)
  data = cpy.random.randint(0, upperLimit, [size, nFields])

  gpu = cuda.to_device(data)
  df = cdf.DataFrame(gpu, columns = ['Q%d'%(i+1) for i in range(nFields)])

  return df

@cuda.jit
def multiplyGPU(in_col, out_col, multiplier):
    i = cuda.grid(1)
    if i < in_col.size:  # boundary guard
        out_col[i] = in_col[i] * multiplier

@cuda.jit
def sumColsGPU(in_col, out_col):
    i = cuda.grid(1)
    if i < in_col.size:  # boundary guard
        out_col[i] = out_col[i] + in_col[i]

@cuda.jit
def subColsGPU(in_col, out_col):
    i = cuda.grid(1)
    if i < in_col.size:  # boundary guard
        out_col[i] = out_col[i] + in_col[i]

def createValidScenariosGPU(n, nData, supportC, C_total, S_total, C_item, S_item):
  upperLimit = 0
  for i in range(nData):
     ul = C_total // C_item[i]
     if (ul > upperLimit):
       upperLimit = ul

  lowRand = 1
  uppRand = nData

  cpy.random.seed(None)
  bias = cpy.random.randint(lowRand,uppRand)
  upperLimit = (upperLimit // bias)

  df=createRandomGPU(nFields=nData, upperLimit=upperLimit, size=n)

  for i in range(nData):
    df['C%d'%(i+1)]=0.0
    df['S%d'%(i+1)]=0.0

  df["CTotal"]=0.0
  df["STotal"]=0.0

  for i in range(nData):
    #Calculanting Total Cost per row
    multiplyGPU.forall(n)(df['Q%d'%(i+1)], df['C%d'%(i+1)], C_item[i])
    sumColsGPU.forall(n)(df['C%d'%(i+1)], df['CTotal'])

    #Calculanting Total Space per row
    multiplyGPU.forall(n)(df['Q%d'%(i+1)], df['S%d'%(i+1)], S_item[i])
    sumColsGPU.forall(n)(df['S%d'%(i+1)], df['STotal'])

  expr = "((CTotal >= @supportC) and (CTotal <= @C_total)) and (STotal<=@S_total)"

  filtered_df = df.query(expr)
  if filtered_df.shape[0]>0:
    print("bias= ", bias,  " - Solucoes Validas = ", filtered_df.shape[0])

  return filtered_df

def calculateMaxParamsGPU(n, nData, vlItems, df):
  for i in range(nData):
    df['V%d'%(i+1)]=0.0

  df["VTotal"]=0.0

  for i in range(nData):
    multiplyGPU.forall(n)(df['Q%d'%(i+1)], df['V%d'%(i+1)], vlItems[i])
    sumColsGPU.forall(n)(df["V%d"%(i+1)], df['VTotal'])

  df["PTotal"] = df["VTotal"] - df["CTotal"]

  return df

def executeGPU(folder, input_file):
  dtIni = datetime.now();
  stats = cdf.Series([]);
  stats = save_statsGPU(stats, 'Start')
  n = 0  # n = Number of samples
  C_total: float = 0.0  # Budget (Cost)
  S_total: float = 0.0  # Total Cubed Space

  # Loading input file
  df_input = cdf.read_csv(input_file, delimiter=";", header=None);

  if ((len(df_input)>0) and (df_input.iloc[0,0] == 'inputCSV')):
    stats=save_statsGPU(stats, "Start")
    data = cdf.DataFrame();

    n = int(df_input.iloc[0,1])
    CTotal = float(df_input.iloc[0,2])
    STotal = float(df_input.iloc[0,3])
    support = float(df_input.iloc[0,4])
    cSupport = CTotal * support

    items = cdf.Series([df_input.iloc[i, 0] for i in range(1, len(df_input))]);
    cItems = cdf.Series((df_input.iloc[i, 1] for i in range(1, len(df_input))), dtype=cpy.float64);
    sItems = cdf.Series((df_input.iloc[i, 2] for i in range(1, len(df_input))), dtype=cpy.float64);
    vItems = cdf.Series((df_input.iloc[i, 3] for i in range(1, len(df_input))), dtype=cpy.float64);

    nData = len(items)

    sim = cdf.DataFrame();
    simTemp = cdf.Series(cpy.zeros(nData, dtype="int32"))
    nDif = n
    while (nDif>0):
      print(nDif)
      simTemp = createValidScenariosGPU(n=(n*10), nData=nData,  supportC=cSupport, C_total=CTotal, S_total=STotal, C_item=cItems, S_item=sItems)

      sim = cdf.concat([sim, simTemp], ignore_index=True)

      sim = sim.drop_duplicates()

      rowsFound = sim.shape[0]
      print(rowsFound)
      if (rowsFound>n):
        sim = sim.head(n)
        nDif = 0
      else:
        nDif = (n-rowsFound)

    sim = calculateMaxParamsGPU(n=n, nData=nData, vlItems=vItems, df=sim)

    sim.to_csv(folder+'GPU/outputs/simGPU_'+str(n)+'.csv')
    st = calculate_statsGPU(sim=sim, n=n)

    stats=save_statsGPU(stats, 'Mean;StdDev;zc;Low_interv;Upper_interv;Error;Tolerance')
    stats=save_statsGPU(stats, (str(st[0])+';'+str(st[1])+';'+str(st[2])+';'+str(st[3])+';'+str(st[4])+';'+str(st[5])+';'+str(st[6])))

    dtFinal = datetime.now();
    stats=save_statsGPU(stats, 'n = '+str(n) +'; Support = '+str(support)+'; Time='+str(dtFinal-dtIni))
    df = cdf.DataFrame(stats);
    df.to_csv(folder+'GPU/stats/statsGPU_'+str(n)+'.csv')

"""## CPU Routines with Threads"""

def save_statsCPU(stats, texto):
  dt = datetime.now()
  st = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
  stats=pd.concat([stats, pd.Series({"DateTime": st, "Msg":texto})], ignore_index=True)
  return stats

def calcular_statsCPU(sim, n):
      mean = sim["PTotal"].mean()

      somaErroQuadrado = 0
      
      ers = ((sim["PTotal"]-mean)**2)
      somaErroQuadrado = (ers.sum())

      if (n>1):
          desvioPadrao = sqrt(somaErroQuadrado/(n-1))
      else:
          desvioPadrao = 1

      zc = 1.96
      erro = (zc*desvioPadrao)/sqrt(n)
      interv_Inferior = mean - erro
      interv_Superior = mean + erro
      tolerancia = abs(erro / mean)

      st = [mean, desvioPadrao, zc, interv_Inferior, interv_Superior, erro, tolerancia]

      return st

def createRandomCPU(nFields, upperLimit, size):
  np.random.seed(None)
  data = np.random.randint(0, upperLimit, [size, nFields])
  df = pd.DataFrame(data, columns = ['Q%d'%(i+1) for i in range(nFields)])

  return df

def multiplyCPU(in_col, out_col, multiplier):
  out_col = in_col * multiplier
  return out_col

def sumColsCPU(in_col, out_col):
  out_col = out_col + in_col
  return out_col

def createValidScenariosCPU(index, n, nData, supportC, C_total, S_total, C_item, S_item):
  global simItems
  upperLimit = 0
  for i in range(nData):
     ul = C_total // C_item[i]
     if (ul > upperLimit):
       upperLimit = ul

  lowRand = 1
  uppRand = nData

  np.random.seed(None)
  bias = np.random.randint(lowRand,uppRand)
  upperLimit = (upperLimit // bias)

  df=createRandomCPU(nFields=nData, upperLimit=upperLimit, size=n)

  for i in range(nData):
    df['C%d'%(i+1)]=0.0
    df['S%d'%(i+1)]=0.0

  df["CTotal"]=0.0
  df["STotal"]=0.0

  for i in range(nData):
    #Calculanting Total Cost per row
    df['C%d'%(i+1)] = multiplyCPU(df['Q%d'%(i+1)], df['C%d'%(i+1)], C_item[i])
    df['CTotal'] = sumColsCPU(df['C%d'%(i+1)], df['CTotal'])

    #Calculanting Total Space per row
    df['S%d'%(i+1)] = multiplyCPU(df['Q%d'%(i+1)], df['S%d'%(i+1)], S_item[i])
    df['STotal'] = sumColsCPU(df['S%d'%(i+1)], df['STotal'])

  expr = "((CTotal >= @supportC) and (CTotal <= @C_total)) and (STotal<=@S_total)"

  filtered_df = df.query(expr)
  
  if filtered_df.shape[0]>0:
    print("index=", index, " - Solucoes Validas CPU= ", filtered_df.shape[0])
    simItems = pd.concat([simItems, filtered_df], ignore_index=True)

  return filtered_df

def calculateMaxParamsCPU(n, nData, vlItems, df):
  for i in range(nData):
    df['V%d'%(i+1)]=0.0

  df["VTotal"]=0.0

  for i in range(nData):
    df['V%d'%(i+1)] = multiplyCPU(df['Q%d'%(i+1)], df['V%d'%(i+1)], vlItems[i])
    df['VTotal'] = sumColsCPU(df["V%d"%(i+1)], df['VTotal'])

  df["PTotal"] = df["VTotal"] - df["CTotal"]

  return df

def runThreads(nThreads, n, nData, supportC, C_total, S_total, C_item, S_item):
    #Calculating interactions per thread - Error correction sended to last thread
    interacoesThread = list()
    nDif = 0
    totThread = (n // nThreads)
    for index in range(nThreads):
        nDif += totThread
        if ((index == (nThreads-1)) and (nDif != n)): # Error correction - if needed
            interacoesThread.append(totThread+(n-nDif))
        else:
            interacoesThread.append(totThread)

    threads = list()
    for index in range(nThreads):
        x = threading.Thread(target=createValidScenariosCPU, args=(index, interacoesThread[index], nData, supportC, C_total, S_total, C_item, S_item))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()

def executeCPU(folder, input_file):
  global simItems

  dtIni = datetime.now();
  stats = pd.DataFrame([]);
  stats = save_statsCPU(stats, 'Start')
  n = 0  # n = Amostras requeridas
  C_total: float = 0.0  # Budget (Cost)
  S_total: float = 0.0  # Cubed Total Space

  # Loading input files
  df_input = pd.read_csv(input_file, delimiter=";", header=None);

  if ((len(df_input)>0) and (df_input.iloc[0,0] == 'inputCSV')):
    stats=save_statsCPU(stats, "Inicio")
    data = pd.DataFrame();

    n = int(df_input.iloc[0,1])
    CTotal = float(df_input.iloc[0,2])
    STotal = float(df_input.iloc[0,3])
    support = float(df_input.iloc[0,4])
    nThreads = int(df_input.iloc[0,5])

    cSupport = CTotal * support

    items = pd.Series([df_input.iloc[i, 0] for i in range(1, len(df_input))]);
    cItems = pd.Series((df_input.iloc[i, 1] for i in range(1, len(df_input))), dtype=np.float64);
    sItems = pd.Series((df_input.iloc[i, 2] for i in range(1, len(df_input))), dtype=np.float64);
    vItems = pd.Series((df_input.iloc[i, 3] for i in range(1, len(df_input))), dtype=np.float64);

    nData = len(items)

    sim = pd.DataFrame();
    #simItems = pd.Series(np.zeros(nData, dtype="int32"))
    simItems = pd.DataFrame();
    nDif = n
    while (nDif>0):
      print(nDif)

      runThreads(nThreads=nThreads, n=(n*10), nData=nData,  supportC=cSupport, C_total=CTotal, S_total=STotal, C_item=cItems, S_item=sItems);

      sim = pd.concat([sim, simItems], ignore_index=True)
      sim = sim.drop_duplicates()

      rowsFound = sim.shape[0]
      if (rowsFound>n):
        sim = sim.head(n)
        nDif = 0
      else:
        nDif = (n-rowsFound)

    sim = calculateMaxParamsCPU(n=n, nData=nData, vlItems=vItems, df=sim)

    sim.to_csv(folder+'CPU/outputs/simCPU_n'+str(n)+'_threads'+str(nThreads)+'.csv', sep=";")
    st = calcular_statsCPU(sim=sim, n=n)

    stats=save_statsCPU(stats, 'Mean;StdDev;zc;Low_interv;Upper_interv;Error;Tolerance')
    stats=save_statsCPU(stats, (str(st[0])+';'+str(st[1])+';'+str(st[2])+';'+str(st[3])+';'+str(st[4])+';'+str(st[5])+';'+str(st[6])))

    dtFinal = datetime.now();
    stats=save_statsCPU(stats, 'n='+str(n) +'; Support = '+str(support)+'; Time='+str(dtFinal-dtIni))
    df = pd.DataFrame(stats);
    df.to_csv(folder+'CPU/stats/statsCPU_n'+str(n)+'_threads'+str(nThreads)+'.csv', sep=";")

    print("Mean = ", st[0])

"""# Execution Routine"""

def executeFiles():
  # Setting input folder
  folder = "/home/rafael/simbot/"

  #Processing GPU Files
  files = glob.glob(folder+"GPU/inputs/inputGPU*.csv")
  if (len(files)>0):
    for input_file in files:
      executeGPU(folder, input_file)

  files = glob.glob(folder+"CPU/inputs/inputCPU*.csv")
  if (len(files)>0):
    for input_file in files:
      executeCPU(folder, input_file)


executeFiles()


