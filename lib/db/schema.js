// Database schema — table definitions + initialization
// Phase 8 first version: no adjustment, store raw data directly

// Kline table columns (monthly/weekly/daily/60min share same structure, only table name differs)
const KLINE_COLUMNS = `
  code TEXT NOT NULL,
  date TEXT NOT NULL,
  open REAL,
  close REAL,
  high REAL,
  low REAL,
  volume REAL,
  amount REAL,
  amplitude REAL,
  change_percent REAL,
  change_amount REAL,
  turnover_rate REAL,
  adjust REAL DEFAULT 1.0,
  source TEXT DEFAULT NULL,
  PRIMARY KEY (code, date)
`;

const STOCKS_DDL = `
CREATE TABLE IF NOT EXISTS stocks (
  code TEXT PRIMARY KEY,
  market TEXT NOT NULL,
  name TEXT NOT NULL,
  listing_date TEXT,
  delisted INTEGER DEFAULT 0,
  industry TEXT,
  last_updated TEXT
);
`;

const MONTHLY_DDL = `CREATE TABLE IF NOT EXISTS monthly_klines (${KLINE_COLUMNS});`;
const WEEKLY_DDL = `CREATE TABLE IF NOT EXISTS weekly_klines (${KLINE_COLUMNS});`;
const DAILY_DDL = `CREATE TABLE IF NOT EXISTS daily_klines (${KLINE_COLUMNS});`;
const MIN60_DDL = `CREATE TABLE IF NOT EXISTS kline_60min (${KLINE_COLUMNS});`;

const INDEXES = [
  'CREATE INDEX IF NOT EXISTS idx_monthly_code ON monthly_klines(code);',
  'CREATE INDEX IF NOT EXISTS idx_monthly_date ON monthly_klines(date);',
  'CREATE INDEX IF NOT EXISTS idx_weekly_code ON weekly_klines(code);',
  'CREATE INDEX IF NOT EXISTS idx_weekly_date ON weekly_klines(date);',
  'CREATE INDEX IF NOT EXISTS idx_daily_code ON daily_klines(code);',
  'CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_klines(date);',
  'CREATE INDEX IF NOT EXISTS idx_60min_code ON kline_60min(code);',
  'CREATE INDEX IF NOT EXISTS idx_60min_date ON kline_60min(date);',
];

const ADJUST_EVENTS_DDL = `
CREATE TABLE IF NOT EXISTS adjust_events (
  code TEXT NOT NULL,
  date TEXT NOT NULL,
  type TEXT NOT NULL,
  factor REAL,
  description TEXT,
  PRIMARY KEY (code, date, type)
);
`;

// Shenwan Level-1 industry dictionary (legulegu one-time snapshot)
const INDUSTRIES_DDL = `
CREATE TABLE IF NOT EXISTS industries (
  industry_code TEXT PRIMARY KEY,
  industry_name TEXT NOT NULL,
  source TEXT DEFAULT 'sw_l1',
  snapshot_date TEXT
);
`;

// Industry constituent mapping (HS300 scope only)
const STOCK_INDUSTRY_MAPPING_DDL = `
CREATE TABLE IF NOT EXISTS stock_industry_mapping (
  stock_code TEXT NOT NULL,
  industry_code TEXT NOT NULL,
  stock_name TEXT,
  market_cap REAL,
  shares_outstanding REAL,
  snapshot_date TEXT,
  PRIMARY KEY (stock_code, industry_code)
);
`;

// HS300 internal market-cap-weighted sector klines (constituent synthesis)
const HS300_SECTOR_KLINES_DDL = `
CREATE TABLE IF NOT EXISTS hs300_sector_klines (
  sector_code TEXT NOT NULL,
  period TEXT NOT NULL,
  date TEXT NOT NULL,
  open REAL,
  close REAL,
  high REAL,
  low REAL,
  volume REAL,
  amount REAL,
  amplitude REAL,
  change_percent REAL,
  change_amount REAL,
  turnover_rate REAL,
  member_count INTEGER,
  source TEXT DEFAULT 'composite',
  PRIMARY KEY (sector_code, period, date)
);
`;

const PERIOD_TABLES = {
  monthly: 'monthly_klines',
  weekly: 'weekly_klines',
  daily: 'daily_klines',
  '60min': 'kline_60min',
};

/** Execute all table creation + index creation */
export function initSchema(db) {
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');

  // create tables
  db.exec(STOCKS_DDL);
  db.exec(MONTHLY_DDL);
  db.exec(WEEKLY_DDL);
  db.exec(DAILY_DDL);
  db.exec(MIN60_DDL);
  db.exec(ADJUST_EVENTS_DDL);
  db.exec(INDUSTRIES_DDL);
  db.exec(STOCK_INDUSTRY_MAPPING_DDL);
  db.exec(HS300_SECTOR_KLINES_DDL);

  // create indexes
  for (const idx of INDEXES) {
    db.exec(idx);
  }
}

/** Get table name by period */
export function tableForPeriod(period) {
  const t = PERIOD_TABLES[period];
  if (!t) throw new Error(`unknown period: ${period}`);
  return t;
}

/** All supported periods */
export { PERIOD_TABLES };
