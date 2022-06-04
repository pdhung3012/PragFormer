import glob
import sys, os
import operator

from tree_sitter import Language, Parser
sys.path.append(os.path.abspath(os.path.join('../..')))
sys.path.append(os.path.abspath(os.path.join('../../../')))
from UtilFunctions import createDirIfNotExist
from tree_sitter import Language, Parser
import pygraphviz as pgv
import pylab,traceback
from pyparsing import OneOrMore, nestedExpr

def getJsonDict(fpCPP,fpDotGraphAllText,fpDotGraphAllImage,fpDotGraphSimplifyText,fpDotGraphSimplifyImage,parser,nlpObj,offsetContext,isSaveGraph):
    dictJson=None
    g=None
    try:
        f1 = open(fpCPP, 'r')
        strCode = f1.read()
        arrCodes=strCode.split('\n')
        f1.close()
        tree=parser.parse(bytes(strCode, 'utf8'))
        cursor = tree.walk()
        node = cursor.node

        indexComment=-1
        for i in range(0,len(arrCodes)):
            strItem=arrCodes[i].strip()
            if strItem.startswith('//'):
                indexComment=i
                break
        # print('a comment here {}'.format(arrCodes[indexComment]))
        startIndex=indexComment-offsetContext
        endIndex=indexComment+offsetContext

        lstIds=[]
        dictJson=walkTreeAndReturnJSonObject(node,arrCodes,lstIds,nlpObj)

        if isSaveGraph:
            graph = pgv.AGraph(directed=True)
            dictLabel = {}
            dictFatherLabel={}
            getDotGraph(dictJson, dictLabel,dictFatherLabel, graph)
            graph.write(fpDotGraphAllText)
            graph.layout(prog='dot')
            graph.draw(fpDotGraphAllImage)
            # print('draw graph here {}'.format(fpDotGraphAllImage))

            startLine = indexComment - offsetContext
            endLine = indexComment + offsetContext
            simpleGraph = copyGraphWithinLineIndex(graph,dictFatherLabel, startLine, endLine)
            simpleGraph.write(fpDotGraphSimplifyText)
            simpleGraph.layout(prog='dot')
            simpleGraph.draw(fpDotGraphSimplifyImage)
    except:
        dictJson=None
        traceback.print_exc()
    return dictJson

def walkTreeAndReturnJSonObject(node,arrCodes,listId):
    dictJson={}
    strType=str(node.type)
    dictJson['type']=strType
    dictJson['id'] = len(listId)+1
    listId.append(len(listId)+1)
    strStart=str(node.start_point)
    strEnd = str(node.end_point)
    arrStart = strStart.split(',')
    arrEnd = strEnd.split(',')
    startLine = int(arrStart[0].replace('(', ''))
    startOffset = int(arrStart[1].replace(')', ''))
    endLine = int(arrEnd[0].replace('(', ''))
    endOffset = int(arrEnd[1].replace(')', ''))
    dictJson['startLine']=startLine
    dictJson['startOffset'] = startOffset
    dictJson['endLine'] = endLine
    dictJson['endOffset'] = endOffset

    # if strType!='translation_unit' and endLine<33:
    #     return dictJson
    listChildren=node.children

    if listChildren is not None and len(listChildren)>0:
        dictJson['children'] = []
        for i in range(0,len(listChildren)):

            arrChildEnd = str(listChildren[i].end_point).split(',')
            endChildLine = int(arrChildEnd[0].replace('(', ''))
            if endChildLine>=33:
                childNode = walkTreeAndReturnJSonObject(listChildren[i], arrCodes, listId)
                dictJson['children'].append(childNode)
    return dictJson