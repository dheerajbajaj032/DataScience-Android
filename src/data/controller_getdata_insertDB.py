import sqlite3
from getdata import GetData
from database import DatabaseClass


class ControllerClass:

    def __init__(self, dbname):
        self.databaseobj = DatabaseClass(dbname=dbname)
        self.databaseobj.createTable()

    def getdata_insertdb(self, pname, id):
        try:
            data_dict = GetData().App_Summary(pname)
            print "Dheeraj" + str(data_dict)
            code = data_dict['Code']
            nativeheap = data_dict['Native Heap']
            system = data_dict['System']
            private_others = data_dict['Private Other']
            graphics = data_dict['Graphics']
            java_heap = data_dict['Java Heap']
            stack = data_dict['Stack']
            self.databaseobj.insertTable(testid=id, Code=code, Native_Heap=nativeheap,
                                         System=system, Private_Others=private_others,
                                         Graphics=graphics, Java_Heap=java_heap, Stack=stack)
        except Exception as e:
            print e

        #self.databaseobj.closeDB()

    def fetch_Dataframe(self):
        return self.databaseobj.fetchrows()

    def closeDB(self):
        return self.databaseobj.closeDB()