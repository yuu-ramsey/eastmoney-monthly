// Uninstall Native Messaging Host
import { execSync } from 'node:child_process';

const REG_EXE = 'C:\\Windows\\System32\\reg.exe';

const HOST_NAME = 'com.eastmoney_ai.sync';
const REG_KEY = `HKEY_CURRENT_USER\\Software\\Google\\Chrome\\NativeMessagingHosts\\${HOST_NAME}`;

try {
  execSync(`${REG_EXE} delete "${REG_KEY}" /f`, { stdio: 'pipe' });
  console.log(`Removed from registry: ${REG_KEY}`);
  console.log('');
  console.log('Please restart Chrome for changes to take effect.');
} catch (err) {
  if (err.message.includes('unable to find') || err.message.includes('not found')) {
    console.log('Registry key does not exist, nothing to uninstall.');
  } else {
    console.error('Uninstall failed:', err.message);
    process.exit(1);
  }
}
