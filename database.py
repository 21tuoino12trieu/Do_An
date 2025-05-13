import sqlite3
from config import DATABASE_PATH

class SQLDatabase:
    def __init__(self):
        self.db_path = DATABASE_PATH

    def connect(self):
        return sqlite3.connect(self.db_path)

    def execute_query(self,query,params=()):
        connection = self.connect()
        cursor = connection.cursor()

        try:
            cursor.execute(query,params)
            results = cursor.fetchall()
            return results

        except Exception as e:
            print(f"Database error: {e}")
            return None
        finally:
            connection.close()