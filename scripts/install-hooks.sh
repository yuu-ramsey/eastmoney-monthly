#!/bin/sh
# 安装 git pre-commit hook —— 防 API key 意外提交
HOOK_DIR=".git/hooks"
HOOK_FILE="$HOOK_DIR/pre-commit"

mkdir -p "$HOOK_DIR"

cat > "$HOOK_FILE" << 'HOOK_EOF'
#!/bin/sh
# 防 API key 意外提交
if git diff --cached | grep -E "sk-[a-zA-Z0-9]{32,}|DEEPSEEK_API_KEY\s*=\s*['\"]sk-|ANTHROPIC_API_KEY\s*=\s*['\"]sk-"; then
  echo "❌ API key detected in staged files. Commit blocked."
  echo "   Remove real API keys before committing."
  exit 1
fi

# 防 .env 误提交（.env.example 除外）
ENV_FILES=$(git diff --cached --name-only | grep -E "^\.env$|^\.env\." | grep -v ".env.example")
if [ -n "$ENV_FILES" ]; then
  echo "❌ .env file staged for commit. Commit blocked."
  echo "   Files: $ENV_FILES"
  echo "   Use .env.example as template and keep real .env local."
  exit 1
fi

exit 0
HOOK_EOF

chmod +x "$HOOK_FILE"
echo "✅ pre-commit hook installed to $HOOK_FILE"
