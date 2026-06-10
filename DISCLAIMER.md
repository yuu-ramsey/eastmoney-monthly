# 免责声明 / Disclaimer

## 中文

1. **非投资建议**。本项目（包括 Chrome 插件、命令行工具、研究脚本及其输出的任何分析文本、信号、状态标记）仅用于技术研究、学术学习与软件工程演示，**不构成任何形式的投资建议、买卖推荐或金融服务**。所有分析输出均为条件式陈述（"若 X 则验证 Y"），不应被解读为操作指令。

2. **LLM 输出风险**。本项目使用大语言模型生成解读文本。LLM 可能产生**数据幻觉**（如错误的价格、指标数值）、过时信息或逻辑错误。任何关键数值请以交易所及行情软件的官方数据为准。

3. **信号验证状态的含义**。界面与文档中的 verified / anti_signal / pending 标记仅描述该信号在**本项目特定历史样本与检验流程**下的统计表现，不代表未来有效性，亦不构成推荐。历史回测存在过拟合、幸存者偏差、多重检验等固有风险，即使本项目已尽力控制。

4. **数据来源**。本项目通过 akshare、baostock 及东方财富等公开渠道获取行情与财务数据，仅用于个人学习与研究。数据的准确性、完整性、时效性由原始数据提供方负责。本仓库**不再分发任何原始行情数据**；使用者应自行获取数据并遵守相应数据源的使用条款。

5. **无担保**。本软件按 MIT 许可证"按原样"（AS IS）提供，不附带任何明示或默示的担保。开发者及贡献者对因使用本项目造成的任何直接或间接损失（包括但不限于投资亏损、数据丢失）不承担任何责任。

6. **非执业资质**。本项目作者不具备证券投资咨询执业资格，本项目亦非证券投资咨询服务。使用者基于本项目输出做出的任何投资决策，风险完全自负。

7. **无关联声明**。本项目为独立开源项目，与东方财富信息股份有限公司及其关联方无任何关联、授权或合作关系。

8. **隐私**。本项目不收集、不上传任何用户个人信息。LLM 分析请求由用户使用**自己的 API Key** 直接发往所选服务商（Anthropic / DeepSeek），数据流向受所选服务商的隐私政策约束。

## English (Summary)

This project (Chrome extension, CLI tools, research scripts, and all generated analyses/signals) is for **technical research, academic learning, and software engineering demonstration only**. It does **not** constitute investment advice, trading recommendations, or financial services of any kind.

- LLM-generated text may contain hallucinated numbers or outdated information; always verify against official market data.
- Signal statuses (verified / anti_signal / pending) describe statistical performance under this project's specific historical samples and validation pipeline only; they imply nothing about future performance.
- Market data is fetched from public sources (akshare, baostock, Eastmoney) for personal research use; this repository redistributes **no raw market data**.
- Provided "AS IS" under the MIT License, without warranty of any kind. The authors hold no securities advisory license, are not affiliated with Eastmoney, and accept no liability for any losses arising from the use of this project.
- No user data is collected; LLM requests go directly from your machine to your chosen provider using your own API key.
