// 卸载 Native Messaging Host
import { execSync } from 'node:child_process';

const REG_EXE = 'C:\\Windows\\System32\\reg.exe';

const HOST_NAME = 'com.eastmoney_ai.sync';
const REG_KEY = `HKEY_CURRENT_USER\\Software\\Google\\Chrome\\NativeMessagingHosts\\${HOST_NAME}`;

try {
  execSync(`${REG_EXE} delete "${REG_KEY}" /f`, { stdio: 'pipe' });
  console.log(`已从注册表删除: ${REG_KEY}`);
  console.log('');
  console.log('请重启 Chrome 使更改生效。');
} catch (err) {
  if (err.message.includes('unable to find') || err.message.includes('not found')) {
    console.log('注册表项不存在，无需卸载。');
  } else {
    console.error('卸载失败:', err.message);
    process.exit(1);
  }
}
