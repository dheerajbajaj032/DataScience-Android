import sqlite3
from getdata import GetData
from database import DatabaseClass


class ControllerClass:

    def __init__(self, dbname):
        self.databaseobj = DatabaseClass(dbname=dbname)
        self.databaseobj.createTable()

    def getdata_insertdb(self, pname, id):
        data_dict = GetData().getmeminfo(pname)
        pss = data_dict['Native_Pss']
        nativedirty = data_dict['Native_Private_Dirty']
        heapalloc = data_dict['Native_Heap_Alloc']
        heapfree = data_dict['Native_Heap_Free']
        codepss = data_dict['code_Pss']
        codedirty = data_dict['code_Private_Dirty']
        self.databaseobj.insertTable(testid=id, pss=pss, nativedirty=nativedirty, heapalloc=heapalloc,
                                     heapfree=heapfree, codepss=codepss, codedirty=codedirty)
        self.databaseobj.closeDB()

    def fetch_Dataframe(self):
        return self.databaseobj.fetchrows()

    def closeDB(self):
        return self.databaseobj.closeDB()