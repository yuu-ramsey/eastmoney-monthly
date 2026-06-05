// Database connection management — singleton, lazy initialization
// Database file location: .eastmoney-ai/db/klines.sqlite

import Database from 'better-sqlite3';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { initSchema } from './schema.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const DB_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'db');
const DB_PATH = path.join(DB_DIR, 'klines-v2.sqlite');

let _db = null;

/** Get database instance (lazy creation, auto-create tables) */
export function getDb() {
  if (_db) return _db;

  if (!fs.existsSync(DB_DIR)) fs.mkdirSync(DB_DIR, { recursive: true });

  const isNew = !fs.existsSync(DB_PATH);
  _db = new Database(DB_PATH);

  if (isNew) {
    initSchema(_db);
  }

  return _db;
}

/** 关闭连接（程序退出时调用） */
export function closeDb() {
  if (_db) {
    _db.close();
    _db = null;
  }
}

/** 获取数据库文件路径（仅用于状态显示） */
export function getDbPath() {
  return DB_PATH;
}
