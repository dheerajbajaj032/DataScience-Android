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
                     Code           INT    NOT NULL,
                     Native_Heap           INT    NOT NULL,
                     System           INT    NOT NULL,
                     Private_Others           INT    NOT NULL,
                     Graphics           INT    NOT NULL,
                     Java_Heap           INT    NOT NULL,
                     Stack           INT    NOT NULL
                     );''')
            print "Table created successfully"
        except Exception as e:
            print "Table already exists" + str(e)

    def insertTable(self, testid, Code, Native_Heap, System, Private_Others, Graphics, Java_Heap, Stack):
        global cur
        sql = '''INSERT INTO ANDROID (TESTId,Code, Native_Heap, System, Private_Others, Graphics, Java_Heap, Stack) \
              VALUES (?,?,?,?,?,?,?,?)'''
        query = (testid, Code, Native_Heap, System, Private_Others, Graphics, Java_Heap, Stack);
        cur = self.conn.cursor()
        cur.execute(sql, query)
        self.conn.commit()
        print "Record created successfully"
        print cur.lastrowid
        # print self.conn.execute('''SELECT * FROM ANDROID''')

    def fetchrows(self):
        df = pd.read_sql_query("SELECT * FROM ANDROID", self.conn)
        df.set_index('TESTId')
        df.to_csv("ANDROID.csv")
        return df

    def closeDB(self):
        self.conn.close()
        print "Database closed successfully"
