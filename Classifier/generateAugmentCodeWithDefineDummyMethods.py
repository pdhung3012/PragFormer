import glob
import os

from UtilFunctions import createDirIfNotExist

fopRoot='/home/hungphd/git/Open_OMP/augmentCodesAndLogs/'
fopInput='/home/hungphd/git/Open_OMP/augmentCodesAndLogs/Parallel_Augmented_v1/'
fopOutput='/home/hungphd/git/Open_OMP/augmentCodesAndLogs/Parallel_Augmented_v2/'
createDirIfNotExist(fopOutput)
dictIncludeCount={}
lstFpCFiles=glob.glob(fopInput+'*.c')
for i in range(0,len(lstFpCFiles)):
    fpItem=lstFpCFiles[i]
    f1=open(fpItem,'r')
    arrInCode=f1.read().strip().split('\n')
    f1.close()
    fnName = os.path.basename(fpItem)
    lineEndInclude=0
    while(lineEndInclude<len(arrInCode)):
        strStrip=arrInCode[lineEndInclude].strip()
        if strStrip.startswith('#include'):
            break
        lineEndInclude+=1
    lstIncludes=[]
    while (lineEndInclude < len(arrInCode)):
        strStrip = arrInCode[lineEndInclude].strip()
        if not strStrip.startswith('#include'):
            break
        lstIncludes.append(strStrip)
        if strStrip not in dictIncludeCount.keys():
            dictIncludeCount[strStrip]=1
        else:
            dictIncludeCount[strStrip]=dictIncludeCount[strStrip]+1
        lineEndInclude += 1
    lstOutCode=[]
    for j in range(0,len(arrInCode)):
        if j==0 and len(lstIncludes)==0:
            lstOutCode.append('#include <stdio.h>\nint dummyMethod1();\nint dummyMethod2();\nint dummyMethod3();\nint dummyMethod4();')
            # input('go here {}'.format(fnName))
        elif j==lineEndInclude:
            lstOutCode.append('int dummyMethod1();\nint dummyMethod2();\nint dummyMethod3();\nint dummyMethod4();')
        lstOutCode.append(arrInCode[j])

    f1=open(fopOutput+fnName,'w')
    f1.write('\n'.join(lstOutCode))
    f1.close()

dictIncludeCount=dict(sorted(dictIncludeCount.items(), key=lambda item: item[1],reverse=True))

lstStr=[]
for key in dictIncludeCount.keys():
    lstStr.append('{}'.format(key,dictIncludeCount[key]))
f1=open(fopRoot+'stat_include.txt','w')
f1.write('\n'.join(lstStr))
f1.close()

