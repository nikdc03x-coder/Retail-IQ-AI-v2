// Automatically generated start script with dynamic port selection
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const net = require('net');

// Locate python executable from the virtual environment (Windows or Unix)
const venvPythonWin = path.join(__dirname, '.venv', 'Scripts', 'python.exe');
const venvPythonUnix = path.join(__dirname, '.venv', 'bin', 'python');
let pythonBin = 'python';
if (fs.existsSync(venvPythonWin)) {
  pythonBin = venvPythonWin;
} else if (fs.existsSync(venvPythonUnix)) {
  pythonBin = venvPythonUnix;
}

// Arguments passed after "npm run dev"
const userArgs = process.argv.slice(2);

// Helper to locate a free TCP port starting from a desired port
function getFreePort(startPort) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.unref();
    server.on('error', () => {
      // Port in use – try the next one recursively
      resolve(getFreePort(startPort + 1));
    });
    server.listen(startPort, () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

(async () => {
  let args;
  // If the user explicitly passes a Node‑module flag (e.g. "-m pytest"), forward it unchanged
  if (userArgs.length > 0 && userArgs[0] === '-m') {
    args = userArgs;
  } else {
    // Determine desired port (env overrides default 8501) and ensure it is free
    const defaultPort = parseInt(process.env.STREAMLIT_SERVER_PORT) || 8501;
    const freePort = await getFreePort(defaultPort);
    args = [
      '-m',
      'streamlit',
      'run',
      'app.py',
      '--server.headless',
      'true',
      '--server.port',
      freePort.toString(),
      ...userArgs,
    ];
    console.log(`\x1b[36m[RetailIQ AI]\x1b[0m Starting application via ${pythonBin} on port ${freePort}...`);
  }

  const child = spawn(pythonBin, args, { stdio: 'inherit' });
  child.on('exit', (code) => process.exit(code ?? 0));
})();
