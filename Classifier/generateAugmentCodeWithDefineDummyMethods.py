import glob
import os

from UtilFunctions import createDirIfNotExist

fopInput='/home/hungphd/git/Open_OMP/augmentCodesAndLogs/Parallel_Augmented_v1/'
fopOutput='/home/hungphd/git/Open_OMP/augmentCodesAndLogs/Parallel_Augmented_v2/'
createDirIfNotExist(fopOutput)

lstFpCFiles=glob.glob(fopInput+'*.c')
for i in range(0,len(lstFpCFiles)):
    fpItem=lstFpCFiles[i]
    f1=open(fpItem,'r')
    arrInCode=f1.read().strip().split('\n')
    lineEndInclude=0
    while(lineEndInclude<len(arrInCode)):
        strStrip=arrInCode[lineEndInclude].strip()
        if strStrip.startswith('#include'):
            break
        lineEndInclude+=1
    while (lineEndInclude < len(arrInCode)):
        strStrip = arrInCode[lineEndInclude].strip()
        if not strStrip.startswith('#include'):
            break
        lineEndInclude += 1
    lstOutCode=[]
    for j in range(0,len(arrInCode)):
        if j==lineEndInclude:
            lstOutCode.append('int dummyMethod1();\nint dummyMethod2();')
        lstOutCode.append(arrInCode[j])
    fnName=os.path.basename(fpItem)
    f1=open(fopOutput+fnName,'w')
    f1.write('\n'.join(lstOutCode))
    f1.close()


