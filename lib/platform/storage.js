// Storage abstraction layer — browser extension uses chrome.storage.local, Node uses files
// Static imports only for env-agnostic parts; Node modules loaded dynamically to avoid browser errors

// Environment detection (no module dependency, works in browser/Node)
const isNodeEnv = typeof globalThis.chrome === 'undefined'
  || typeof globalThis.chrome?.storage?.local === 'undefined';

// ---- Browser implementation ----

function getFromChrome(keys) {
  return new Promise((resolve) => {
    chrome.storage.local.get(keys, resolve);
  });
}

function setToChrome(items) {
  return new Promise((resolve) => {
    chrome.storage.local.set(items, resolve);
  });
}

function removeFromChrome(keys) {
  return new Promise((resolve) => {
    chrome.storage.local.remove(keys, () => resolve());
  });
}

// ---- Node 文件实现（惰性初始化） ----

let _fs = null;
let _path = null;
let _storageDir = null;

async function _ensureNodeModules() {
  if (_fs) return;
  // 仅在 Node 环境动态导入，浏览器不会走到这里
  const [fsMod, pathMod, urlMod] = await Promise.all([
    import('node:fs'),
    import('node:path'),
    import('node:url'),
  ]);
  _fs = fsMod;
  _path = pathMod;

  const __dirname = _path.dirname(urlMod.fileURLToPath(import.meta.url));
  const projectDir = _path.resolve(__dirname, '..', '..');
  _storageDir = _path.join(projectDir, '.eastmoney-ai', 'storage');
}

function _ensureDir() {
  if (!_fs.existsSync(_storageDir)) _fs.mkdirSync(_storageDir, { recursive: true });
}

function _keyToPath(key) {
  const safeKey = String(key).replace(/[:*?"<>|]/g, '_');
  return _path.join(_storageDir, `${safeKey}.json`);
}

async function getFromFile(keys) {
  await _ensureNodeModules();
  _ensureDir();

  if (keys === null) {
    const result = {};
    try {
      const files = _fs.readdirSync(_storageDir).filter((f) => f.endsWith('.json'));
      for (const f of files) {
        const k = _path.basename(f, '.json').replace(/_/g, ':');
        try {
          result[k] = JSON.parse(_fs.readFileSync(_path.join(_storageDir, f), 'utf-8'));
        } catch (_) { /* 损坏文件跳过 */ }
      }
    } catch (_) { /* 目录不存在 */ }
    return result;
  }

  const keyList = Array.isArray(keys) ? keys : [keys];
  const result = {};
  for (const k of keyList) {
    const filePath = _keyToPath(k);
    try {
      result[k] = JSON.parse(_fs.readFileSync(filePath, 'utf-8'));
    } catch (_) { /* 文件不存在视为 undefined */ }
  }
  return result;
}

async function setToFile(items) {
  await _ensureNodeModules();
  _ensureDir();
  for (const [k, v] of Object.entries(items)) {
    const filePath = _keyToPath(k);
    _fs.writeFileSync(filePath, JSON.stringify(v, null, 2), 'utf-8');
  }
}

async function removeFromFile(keys) {
  await _ensureNodeModules();
  const keyList = Array.isArray(keys) ? keys : [keys];
  for (const k of keyList) {
    const filePath = _keyToPath(k);
    try {
      if (_fs.existsSync(filePath)) _fs.unlinkSync(filePath);
    } catch (_) { /* ignore */ }
  }
}

// ---- 统一接口 ----

export const get = isNodeEnv ? getFromFile : getFromChrome;
export const set = isNodeEnv ? setToFile : setToChrome;
export const remove = isNodeEnv ? removeFromFile : removeFromChrome;
export { isNodeEnv as isNode };
