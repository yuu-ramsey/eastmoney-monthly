@echo off
REM Use NODE_PATH env var; fallback to node from PATH
if defined NODE_PATH (
    "%NODE_PATH%" "%~dp0\server.js"
) else (
    node "%~dp0\server.js"
)
