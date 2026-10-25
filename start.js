const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const venvPythonWin = path.join(__dirname, '.venv', 'Scripts', 'python.exe');
const venvPythonUnix = path.join(__dirname, '.venv', 'bin', 'python');

let pythonBin = 'python';
if (fs.existsSync(venvPythonWin)) {
  pythonBin = venvPythonWin;
} else if (fs.existsSync(venvPythonUnix)) {
  pythonBin = venvPythonUnix;
}

const userArgs = process.argv.slice(2);
let args;
if (userArgs.length > 0 && userArgs[0] === '-m') {
  args = userArgs;
} else {
  args = ['-m', 'streamlit', 'run', 'app.py', ...userArgs];
}

console.log(`\x1b[36m[RetailIQ AI]\x1b[0m Starting application via ${pythonBin}...`);

const child = spawn(pythonBin, args, {
  stdio: 'inherit'
});

child.on('exit', (code) => {
  process.exit(code ?? 0);
});

