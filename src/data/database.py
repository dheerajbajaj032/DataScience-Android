import sqlite3
import pandas as pd


class DatabaseClass:

    def __init__(self, dbname):
        self.conn = sqlite3.connect(dbname)
        print "Database connected successfully"

    def createTable(self):

        try:
            self.conn.execute('''CREATE TABLE ANDROID
                     (TESTId     TEXT     NOT NULL,
                     Native_Pss           INT    NOT NULL,
                     Native_Private_Dirty           INT    NOT NULL,
                     Native_Heap_Alloc           INT    NOT NULL,
                     Native_Heap_Free           INT    NOT NULL,
                     code_Pss           INT    NOT NULL,
                     code_Private_Dirty           INT    NOT NULL
                     );''')
            print "Table created successfully"
        except Exception as e:
            print "Table already exists"

    def insertTable(self, testid, pss, nativedirty, heapalloc, heapfree, codepss, codedirty):
        global cur
        sql = '''INSERT INTO ANDROID (TESTId,Native_Pss,Native_Private_Dirty,Native_Heap_Alloc,Native_Heap_Free,code_Pss,code_Private_Dirty) \
              VALUES (?,?,?,?,?,?,?)'''
        query = (testid, pss, nativedirty, heapalloc, heapfree, codepss, codedirty);
        cur = self.conn.cursor()
        cur.execute(sql, query)
        self.conn.commit()
        print "Record created successfully"
        print cur.lastrowid
        # print self.conn.execute('''SELECT * FROM ANDROID''')

    def fetchrows(self):
        df = pd.read_sql_query("SELECT * FROM ANDROID", self.conn)
        df.set_index('TESTId')
        #print df
        #df.to_csv("ANDROID.csv")
        return df

    def closeDB(self):
        self.conn.close()
        print "Database closed successfully"
