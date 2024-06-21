import sqlite3

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn):
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS reference_audios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL UNIQUE,
        file_name TEXT NOT NULL,
        file_path TEXT NOT NULL
    );
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)

def insert_reference_audio(conn, user_id, file_name, file_path):
    sql = '''INSERT INTO reference_audios (user_id, file_name, file_path)
             VALUES(?, ?, ?)'''
    cur = conn.cursor()
    cur.execute(sql, (user_id, file_name, file_path))
    conn.commit()
    return cur.lastrowid

# def select_all_reference_audios(conn, user_id):
#     sql = "SELECT file_name, file_path FROM reference_audios WHERE user_id=?"
#     cur = conn.cursor()
#     cur.execute(sql, (user_id,))
#     rows = cur.fetchall()
#     print(rows)
#     return rows

def get_file_name_by_user_id(conn, user_id):
    try:
        c = conn.cursor()
        c.execute("SELECT file_name FROM reference_audios WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        if result:
            return result[0]
        else:
            return None
    except sqlite3.Error as e:
        print(e)
        return None
    
def get_file_path_by_user_id(conn, user_id):
    try:
        c = conn.cursor()
        c.execute("SELECT file_path FROM reference_audios WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        if result:
            return result[0]
        else:
            return None
    except sqlite3.Error as e:
        print(e)
        return None

def select_all_users(conn):
    sql = "SELECT DISTINCT user_id FROM reference_audios"
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    return rows

def save_user(conn, user_id, file_name, file_path):
    sql = '''INSERT INTO reference_audios(user_id, file_name, file_path)
            values(?, ?, ?)'''
    cur = conn.cursor()
    cur.execute(sql, (user_id, file_name, file_path))
    conn.commit()

def update_reference(conn, user_id, file_name, file_path):
    sql = '''UPDATE reference_audios
             SET file_name = ?,
                 file_path = ?
             WHERE user_id = ?'''
    cur = conn.cursor()
    cur.execute(sql, (file_name, file_path, user_id))
    conn.commit()

def insert_user(conn, user_id):
    sql = '''INSERT INTO reference_audios(user_id, file_name, file_path)
             VALUES(?, ?, ?)'''
    cur = conn.cursor()
    cur.execute(sql, (user_id, "", ""))
    conn.commit()
    return cur.lastrowid

def get_path_from_file_name(conn, file_name):
    sql = "SELECT file_path FROM reference_audios WHERE file_name=?"
    cur = conn.cursor()
    cur.execute(sql, (file_name,)) # Comma is put to make it a tuple with one element (not a string).
    row = cur.fetchone()
    return row[0] if row else None
