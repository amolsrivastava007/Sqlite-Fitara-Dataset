import sqlite3

#ESTABLISH CONNECTION
conn = sqlite3.connect('FITARA.db')

#CREATE DB
print ("Opened database successfully");

#EXCECUTE QUERY
conn.execute('CREATE TABLE fitt (doc TEXT, prediction TEXT,ans Text)')
print ("Table created successfully")
conn.close()
