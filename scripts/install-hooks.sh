#!/bin/sh
# Install git pre-commit hook — prevent accidental API key commit
HOOK_DIR=".git/hooks"
HOOK_FILE="$HOOK_DIR/pre-commit"

mkdir -p "$HOOK_DIR"

cat > "$HOOK_FILE" << 'HOOK_EOF'
#!/bin/sh
# Prevent accidental API key commit
if git diff --cached | grep -E "sk-[a-zA-Z0-9]{32,}|DEEPSEEK_API_KEY\s*=\s*['\"]sk-|ANTHROPIC_API_KEY\s*=\s*['\"]sk-"; then
  echo "❌ API key detected in staged files. Commit blocked."
  echo "   Remove real API keys before committing."
  exit 1
fi

# Prevent accidental .env commit (.env.example excluded)
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
