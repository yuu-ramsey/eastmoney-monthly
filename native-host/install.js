// One-click install Native Messaging Host
// Usage: node native-host/install.js <扩展ID>
// 如未提供扩展ID，脚本会提示如何获取

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
  console.log('Usage: node native-host/install.js <扩展ID>');
  console.log('');
  console.log('获取扩展ID:');
  console.log('  1. 打开 chrome://extensions');
  console.log('  2. 右上角开启"开发者模式"');
  console.log('  3. 找到"东方财富月线 AI 分析"扩展');
  console.log('  4. 复制"ID"字段（32 位小写字母）');
  console.log('');
  console.log('例如: node native-host/install.js abcdefghijklmnopqrstuvwxyz123456');
  process.exit(1);
}

// 验证扩展ID格式
if (!/^[a-z]{32}$/.test(extensionId)) {
  console.error('错误: 扩展ID 格式不正确，应为 32 位小写字母');
  process.exit(1);
}

// 1. 写入 manifest（替换占位符）
const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8'));
manifest.path = LAUNCHER_PATH;
manifest.allowed_origins = [`chrome-extension://${extensionId}/`];

const installedManifestPath = path.join(__dirname, 'manifest', 'eastmoney-ai-sync-installed.json');
fs.writeFileSync(installedManifestPath, JSON.stringify(manifest, null, 2), 'utf-8');
console.log(`[1/3] manifest 已生成: ${installedManifestPath}`);

// 2. 写注册表
try {
  execSync(`${REG_EXE} add "${REG_KEY}" /ve /t REG_SZ /d "${installedManifestPath}" /f`, {
    stdio: 'pipe',
  });
  console.log(`[2/3] 注册表已写入: ${REG_KEY}`);
} catch (err) {
  if (err.message.includes('access denied') || err.message.includes('Access is denied')) {
    console.error('[2/3] 错误: 注册表写入权限不足，请以管理员身份运行此脚本');
    console.error('  右键 PowerShell → 以管理员身份运行 → cd 到项目目录 → 重试');
    process.exit(1);
  }
  throw err;
}

// 3. 验证
try {
  const result = execSync(`${REG_EXE} query "${REG_KEY}" /ve`, { stdio: 'pipe' }).toString();
  console.log(`[3/3] 验证通过: ${result.trim()}`);
} catch (err) {
  console.error('[3/3] 验证失败:', err.message);
  process.exit(1);
}

console.log('');
console.log('=== 安装完成 ===');
console.log('');
console.log('下一步:');
console.log('  1. 重启 Chrome（完全关闭后重新打开）');
console.log('  2. 在 chrome://extensions 里重新加载扩展');
console.log('  3. 在东方财富页面做一次分析');
console.log('  4. 检查 .eastmoney-ai/storage/ 是否有文件生成');
console.log('');
console.log('卸载: node native-host/uninstall.js');
