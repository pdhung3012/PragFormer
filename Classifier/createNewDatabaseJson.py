import ast
fpInput='/home/hungphd/git/Open_OMP/database_org.json'
fpOutput='/home/hungphd/git/Open_OMP/database.json'

f1=open(fpInput,'r')
strInputJson=f1.read().strip()
f1.close()

jsonInput=ast.literal_eval(strInputJson)
print(type(jsonInput))
fopOrigin='/home/reemh/LIGHTBITS/DB/'
fopNew='/home/hungphd/git/Open_OMP/'
for key in jsonInput.keys():
    itemJson=jsonInput[key]
    for childKey in itemJson.keys():
        oldStr=itemJson[childKey]
        if type(oldStr) is str:
            itemJson[childKey]=oldStr.replace(fopOrigin,fopNew)

f1=open(fpOutput,'w')
f1.write(str(jsonInput))
f1.close()

