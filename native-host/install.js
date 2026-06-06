// One-click install Native Messaging Host
// Usage: node native-host/install.js <extension-id>
// If no extension ID provided, script will show how to obtain one

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');

const REG_EXE = 'C:\\Windows\\System32\\reg.exe';

const HOST_NAME = 'com.eastmoney_ai.sync';
const MANIFEST_PATH = path.join(__dirname, 'manifest', 'eastmoney-ai-sync.json');
const LAUNCHER_PATH = path.join(__dirname, 'launcher.bat');
const REG_KEY = `HKEY_CURRENT_USER\\Software\\Google\\Chrome\\NativeMessagingHosts\\${HOST_NAME}`;

const extensionId = process.argv[2];

if (!extensionId) {
  console.log('Usage: node native-host/install.js <extension-id>');
  console.log('');
  console.log('How to find your extension ID:');
  console.log('  1. Open chrome://extensions');
  console.log('  2. Enable "Developer mode" (top right)');
  console.log('  3. Find "Eastmoney Monthly AI Analysis" extension');
  console.log('  4. Copy the "ID" field (32 lowercase letters)');
  console.log('');
  console.log('Example: node native-host/install.js abcdefghijklmnopqrstuvwxyz123456');
  process.exit(1);
}

// Validate extension ID format
if (!/^[a-z]{32}$/.test(extensionId)) {
  console.error('Error: Invalid extension ID format, should be 32 lowercase letters');
  process.exit(1);
}

// 1. Write manifest (replace placeholders)
const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8'));
manifest.path = LAUNCHER_PATH;
manifest.allowed_origins = [`chrome-extension://${extensionId}/`];

const installedManifestPath = path.join(__dirname, 'manifest', 'eastmoney-ai-sync-installed.json');
fs.writeFileSync(installedManifestPath, JSON.stringify(manifest, null, 2), 'utf-8');
console.log(`[1/3] manifest generated: ${installedManifestPath}`);

// 2. Write registry
try {
  execSync(`${REG_EXE} add "${REG_KEY}" /ve /t REG_SZ /d "${installedManifestPath}" /f`, {
    stdio: 'pipe',
  });
  console.log(`[2/3] registry entry written: ${REG_KEY}`);
} catch (err) {
  if (err.message.includes('access denied') || err.message.includes('Access is denied')) {
    console.error('[2/3] Error: Insufficient permission to write registry. Run this script as Administrator');
    console.error('  Right-click PowerShell → Run as Administrator → cd to project dir → retry');
    process.exit(1);
  }
  throw err;
}

// 3. Verify
try {
  const result = execSync(`${REG_EXE} query "${REG_KEY}" /ve`, { stdio: 'pipe' }).toString();
  console.log(`[3/3] verification passed: ${result.trim()}`);
} catch (err) {
  console.error('[3/3] verification failed:', err.message);
  process.exit(1);
}

console.log('');
console.log('=== Installation complete ===');
console.log('');
console.log('Next steps:');
console.log('  1. Restart Chrome (close completely then reopen)');
console.log('  2. Reload the extension in chrome://extensions');
console.log('  3. Run one analysis on an Eastmoney stock page');
console.log('  4. Check if files are generated under .eastmoney-ai/storage/');
console.log('');
console.log('Uninstall: node native-host/uninstall.js');
