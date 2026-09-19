import sqlite3
import os

DB_PATH = 'logs/experiment.db'


def init_db():
    os.makedirs('logs', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id TEXT,
        task_id TEXT,
        condition TEXT,
        resource_type TEXT,
        bytes INTEGER,
        utility REAL,
        vpb REAL,
        loaded INTEGER,
        task_confidence REAL,
        task_success INTEGER,
        completion_time REAL
    )''')
    conn.commit()
    conn.close()


def log_decision(entry):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''INSERT INTO logs
        (participant_id, task_id, condition, resource_type, bytes, utility, vpb, loaded, task_confidence, task_success, completion_time)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)''', (
        entry.get('participant_id'),
        entry.get('task_id'),
        entry.get('condition'),
        entry.get('resource_type'),
        entry.get('bytes'),
        entry.get('utility'),
        entry.get('vpb'),
        entry.get('loaded'),
        entry.get('task_confidence'),
        entry.get('task_success'),
        entry.get('completion_time'),
    ))
    conn.commit()
    conn.close()


def get_all_logs():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM logs').fetchall()
    conn.close()
    return [dict(r) for r in rows]
