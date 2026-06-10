# GitHub 上传安全审计（2026-06-10）

> 任务：评估当前项目推送 GitHub 的安全性。全程只读，未修改任何文件、未触碰运行中的进程（baostock 抓取、WAL 活跃的 sqlite 均未动）。

## 背景事实

- 远程仓库：`https://github.com/yuu-ramsey/eastmoney-monthly.git`
- **仓库是公开的**（GitHub API: `private=False`），最后推送 2026-06-06
- `main` 已推送；当前分支 `neutralize-framework` 领先 origin/main 6 个提交（未推送）
- 历史共 111 个提交，438 个被跟踪文件

## 结论一览

| 等级 | 发现 | 状态 |
|------|------|------|
| 🔴 严重 | 根目录乱码文件 `D␺ClaudeProjects…config​tushare_token.txt` 内含 57 字符 tushare token 明文，**不被 .gitignore 覆盖**，`git add .` 会直接暴露到公开仓库 | ✅ 已修复：与 `config/tushare_token.txt` 比对一致后删除，加 `*tushare_token*` 规则 |
| 🟠 高 | `git add .` 会扫入 ~4.4GB 未跟踪文件：`data/kronos-training/` 2.1GB、`models/` 2GB、`data/pead-baostock.sqlite` 118MB（超 GitHub 100MB 硬限制，且是抓取进程正在写的活库） | ✅ 已修复：全部加入 .gitignore，11 个目标 `git check-ignore` 验证通过 |
| 🟡 中 | `.gitignore` 的 `config/` 规则是未提交的工作区改动 | ✅ 已随安全提交落地 |
| 🟡 中 | `data/baostock-klines-cache.json`（~10MB K线）、`data/SwClass2021_stock.xls`（申万分类）已公开，属行情数据再分发的授权灰区 | 已公开，知悉即可 |
| 🟢 干净 | 见下节 | — |

## 已验证干净的项

1. **token 从未进过 git 历史**：对 token 原值全历史 `git grep`（111 commits）零命中
2. **HEAD 与全历史无密钥模式**：`sk-ant-` / `sk-<hex32>` / `set_token(<hex>)` / `Bearer` / `AKIA` / `ghp_` 仅命中 `.env.example` 的占位符 `sk-ant-your-key-here`
3. **无个人信息泄漏**：跟踪文件中无本机用户目录路径、无邮箱、无真实姓名；README 仅含仓库自身 URL
4. `.npmrc` 仅 `ignore-scripts=true`；`native-host/manifest/eastmoney-ai-sync.json` 全部是 PLACEHOLDER
5. `.claude/` 无文件被跟踪，`settings.local.json` 已 ignore
6. `.aris/` 仅 2 个 research-review trace 文件，密钥模式扫描零命中
7. `data/frozen-eval-*.json` 虽被 .gitignore 注释标为 "private info" 但实际已被跟踪并公开——核查内容仅为股票代码清单 + 评测配置，**无真实私密信息**（该注释主要针对 `.eastmoney-ai/` 的预算数据，后者确实未被跟踪）
8. 未推送的 6 个提交（neutralize 框架 + 文档 + baostock 控制变量抓取脚本）：纯代码与文档，脚本不含任何凭据（baostock 匿名登录、tushare 脚本仅注释提及 token）——**推送安全**

## 修复建议（用户批准后已于 2026-06-10 执行，见状态列；第 4、5 条为操作守则）

1. **处理 token 文件**（二选一，建议先做 a 再做 b）：
   - a. 在 `.gitignore` 追加一行 `*tushare_token*`（先堵住 `git add .` 的口子）
   - b. 删除根目录乱码文件（真正的 token 应放 `config/tushare_token.txt`，该目录已被 ignore）。该文件由某次脚本把 `D:\...\config\tushare_token.txt` 路径分隔符丢失后误建，无进程依赖它
   - token 从未上公网，**不强制轮换**；介意的话去 tushare 重置一次成本也低
2. **补充 .gitignore**（防 `git add .` 事故 + GitHub 体积限制）：
   ```gitignore
   *tushare_token*
   models/
   data/kronos-training*/
   data/kronos-compare/
   data/*.sqlite
   data/*.sqlite-shm
   data/*.sqlite-wal
   klines-v2.sqlite
   *.pid
   .aris/
   kronos/finetune_csv/
   ```
3. **commit 现有的 `.gitignore` 工作区改动**（`config/` 行）
4. **抓取进程运行期间不要 `git add .` / `git add -A`**：staging 活跃 WAL 的 sqlite 既会拿到不一致快照，也会撞 100MB 限制被 push 拒绝；提交新工作请显式指定文件
5. 若本意是私有仓库，去 GitHub Settings → Danger Zone 改 visibility（当前为 Public）

## 审计方法（可复现）

- `git ls-files` 全量过目 + 重点文件读取
- `git grep -E <密钥正则> HEAD` 与 `git grep <pattern> $(git rev-list --all)` 全历史扫描
- `git check-ignore -v` 验证 ignore 覆盖
- GitHub 匿名 API 查 visibility
- `git status --porcelain` 枚举 118 个未跟踪条目并测量体积（仅读元数据）
