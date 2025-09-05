import sqlite3

def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    # Drop old table if exists (fresh start)
    c.execute("DROP TABLE IF EXISTS users")

    # Create new users table
    c.execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized with users table (with username)")

if __name__ == "__main__":
    init_db()
