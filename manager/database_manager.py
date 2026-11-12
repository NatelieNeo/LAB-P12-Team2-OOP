import sqlite3
from sqlite3 import Error

class DatabaseManager:
    def __init__(self, db_file):
        """Initialize the DatabaseManager with the path to the database file."""
        self.db_file = db_file
        self.connection = None

    def create_connection(self):
        """Create a database connection to the SQLite database specified by db_file."""
        try:
            self.connection = sqlite3.connect(self.db_file)
            print(f"Connected to database: {self.db_file}")
        except Error as e:
            print(f"Error connecting to database: {e}")

    def close_connection(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

    def execute_query(self, query, params=None):
        """Execute a single query with optional parameters."""
        if self.connection is None:
            print("No database connection. Please create a connection first.")
            return

        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.connection.commit()
            print("Query executed successfully.")
        except Error as e:
            print(f"Error executing query: {e}")

    def execute_script(self, script):
        """Execute a script containing multiple SQL statements."""
        if self.connection is None:
            print("No database connection. Please create a connection first.")
            return

        try:
            cursor = self.connection.cursor()
            cursor.executescript(script)
            self.connection.commit()
            print("Script executed successfully.")
        except Error as e:
            print(f"Error executing script: {e}")


    def fetch_query(self, query, params=None):
        """Fetch results from a query with optional parameters."""
        if self.connection is None:
            print("No database connection. Please create a connection first.")
            return None

        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            results = cursor.fetchall()
            return results
        except Error as e:
            print(f"Error fetching query: {e}")
            return None

    def load_schema(self, schema_file):
        """Load the database schema from a SQL file."""
        if self.connection is None:
            print("No database connection. Please create a connection first.")
            return

        try:
            with open(schema_file, 'r') as file:
                schema_sql = file.read()

            self.execute_script(schema_sql)
            print(f"Schema loaded from {schema_file}.")

        except FileNotFoundError:
            print(f"Schema file {schema_file} not found.")
        except Error as e:
            print(f"Error loading schema: {e}")


if __name__ == "__main__": 
    db = DatabaseManager("data/database.db")
    db.create_connection()

    schema_file = "data/calorie_coach_schema.sql"

    db.load_schema(schema_file)