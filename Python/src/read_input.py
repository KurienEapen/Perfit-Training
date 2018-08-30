import sqlite3
import os
import time


def sqlite_setup(path=None):
	db_path = os.path.join(path, "skeleton.db")
	return sqlite3.connect(db_path)


if __name__ == '__main__':
	conn = sqlite_setup("C:\\Perfit\\")
	cur = conn.cursor()
	while True:
		cur.execute("SELECT * FROM skeleton")
		
		rows = cur.fetchall()
		
		for row in rows:
			print(row)
