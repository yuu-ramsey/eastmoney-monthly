// Native Messaging Host —— 接收 Chrome 扩展发来的数据，写入 .eastmoney-ai/storage/
// 协议：stdin 读 4 字节长度前缀 + JSON，stdout 写相同格式响应
// Chrome 按需启动此进程，不常驻

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const STORAGE_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'storage');

function ensureStorageDir() {
  if (!fs.existsSync(STORAGE_DIR)) fs.mkdirSync(STORAGE_DIR, { recursive: true });
}

// 读取长度前缀的消息
function readMessage() {
  return new Promise((resolve) => {
    const header = Buffer.alloc(4);
    let headerOffset = 0;
    let body = null;
    let bodyLength = 0;
    let bodyOffset = 0;

    function readHeader(chunk) {
      const remaining = 4 - headerOffset;
      const copyLen = Math.min(chunk.length, remaining);
      chunk.copy(header, headerOffset, 0, copyLen);
      headerOffset += copyLen;

      if (headerOffset >= 4) {
        // 读取小端 uint32
        bodyLength = header.readUInt32LE(0);
        if (bodyLength > 1024 * 1024) {
          // 单条消息不超过 1MB
          sendError('消息过大: ' + bodyLength);
          process.exit(1);
        }
        body = Buffer.alloc(bodyLength);
        // 继续处理剩余数据
        const rest = chunk.subarray(copyLen);
        if (rest.length > 0) readBody(rest);
      }
    }

    function readBody(chunk) {
      const remaining = bodyLength - bodyOffset;
      const copyLen = Math.min(chunk.length, remaining);
      chunk.copy(body, bodyOffset, 0, copyLen);
      bodyOffset += copyLen;

      if (bodyOffset >= bodyLength) {
        try {
          resolve(JSON.parse(body.toString('utf-8')));
        } catch (err) {
          sendError('JSON 解析失败: ' + err.message);
          resolve(null);
        }
      }
    }

    const onData = (chunk) => {
      if (!body) {
        readHeader(chunk);
      } else {
        readBody(chunk);
      }
    };

    process.stdin.on('data', onData);
    process.stdin.on('end', () => resolve(null));
    process.stdin.on('error', () => resolve(null));
  });
}

// 写长度前缀的响应
function sendMessage(obj) {
  const json = JSON.stringify(obj);
  const body = Buffer.from(json, 'utf-8');
  const header = Buffer.alloc(4);
  header.writeUInt32LE(body.length, 0);
  process.stdout.write(Buffer.concat([header, body]));
}

function sendError(msg) {
  sendMessage({ type: 'error', message: msg });
}

function sendAck(msg) {
  sendMessage({ type: 'ack', message: msg });
}

// ---- 消息处理 ----

function handleMessage(msg) {
  if (!msg || !msg.type) {
    sendError('缺少 type 字段');
    return;
  }

  switch (msg.type) {
    case 'sync': {
      ensureStorageDir();
      const { payload } = msg;
      if (!payload || !payload.key) {
        sendError('sync 缺少 payload.key');
        return;
      }

      const safeKey = String(payload.key).replace(/[:*?"<>|]/g, '_');
      const filePath = path.join(STORAGE_DIR, `${safeKey}.json`);

      try {
        fs.writeFileSync(filePath, JSON.stringify(payload.value, null, 2), 'utf-8');
        sendAck(`同步成功: ${payload.key}`);
      } catch (err) {
        sendError(`写入失败: ${err.message}`);
      }
      break;
    }

    case 'sync_batch': {
      ensureStorageDir();
      const { items } = msg;
      if (!items || typeof items !== 'object') {
        sendError('sync_batch 缺少 items');
        return;
      }

      let ok = 0;
      let fail = 0;
      for (const [k, v] of Object.entries(items)) {
        const safeKey = String(k).replace(/[:*?"<>|]/g, '_');
        const filePath = path.join(STORAGE_DIR, `${safeKey}.json`);
        try {
          fs.writeFileSync(filePath, JSON.stringify(v, null, 2), 'utf-8');
          ok++;
        } catch (_) {
          fail++;
        }
      }
      sendAck(`批量同步: ${ok} 成功${fail > 0 ? ', ' + fail + ' 失败' : ''}`);
      break;
    }

    case 'query_sector_alpha': {
      handleQuerySectorAlpha(msg).then(sendMessage).catch(err => sendError(err.message));
      return; // 异步响应，不立即 break
    }

    case 'ping': {
      sendMessage({ type: 'pong' });
      break;
    }

    case 'read': {
      ensureStorageDir();
      const key = String(msg.key || '').replace(/[:*?"<>|]/g, '_');
      if (!key) { sendError('read 缺少 key'); break; }
      const filePath = path.join(STORAGE_DIR, `${key}.json`);
      try {
        if (!fs.existsSync(filePath)) {
          sendMessage({ type: 'read_result', key: msg.key, data: null, exists: false });
        } else {
          const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
          sendMessage({ type: 'read_result', key: msg.key, data, exists: true });
        }
      } catch (err) {
        sendError(`读取失败: ${err.message}`);
      }
      break;
    }

    case 'remove': {
      ensureStorageDir();
      const keys = Array.isArray(msg.keys) ? msg.keys : [msg.key];
      for (const k of keys) {
        const safeKey = String(k).replace(/[:*?"<>|]/g, '_');
        const filePath = path.join(STORAGE_DIR, `${safeKey}.json`);
        try { if (fs.existsSync(filePath)) fs.unlinkSync(filePath); } catch (_) { /* ignore */ }
      }
      sendAck('删除完成');
      break;
    }

    default:
      sendError(`未知消息类型: ${msg.type}`);
  }
}

async function handleQuerySectorAlpha(msg) {
  const { code, period = 'monthly', lookback = 12 } = msg;
  if (!code) return { type: 'error', message: '缺少 code' };

  try {
    const { getDb } = await import('../lib/db/connection.js');
    const { calcSectorAlpha } = await import('../lib/sector/alpha.js');
    const db = getDb();
    const result = calcSectorAlpha(db, code, period, lookback);
    return { type: 'sector_alpha', data: result };
  } catch (err) {
    return { type: 'error', message: err.message };
  }
}

// ---- 主循环 ----

async function main() {
  while (true) {
    const msg = await readMessage();
    if (msg === null) break; // stdin 关闭
    handleMessage(msg);
  }
}

main().catch((err) => {
  try { sendError('进程异常: ' + err.message); } catch (_) { /* ignore */ }
  process.exit(1);
});
