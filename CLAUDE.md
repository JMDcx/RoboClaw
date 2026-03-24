# RoboClaw Agent 开发指南

> 相关文档：
> - 产品愿景：[`docs/product-vision.md`](docs/product-vision.md)
> - 架构设计：[`docs/architecture.md`](docs/architecture.md)
> - 架构对比（LeRobot / dimos / RoboClaw）：[`docs/architecture-comparison.md`](docs/architecture-comparison.md)

---

运用第一性原理思考，拒绝经验主义和路径盲从，不要假设我完全清楚目标，保持审慎，从原始需求和问题出发，若目标模糊请停下和我讨论，若目标清晰但路径非最优，请直接建议更短、更低成本的办法。

---

## 工作规范

### 多角度思考（Codex 协作）

当用户要求从多角度思考、需要更全面的方案时，根据当前阶段调用对应的 Codex skill：

- **规划阶段**：用 `/codex-plan` 让 Codex 出一份独立的实现方案，对比自己的思路后整合。
- **编码阶段**：用 `/codex-dispatch` 将子任务分发给 Codex 并行编写，Codex 可在 worktree 中独立工作；如果 worktree 不方便则一起改同一份代码。
- **审查阶段**：用 `/codex-review` 让 Codex 从新视角审查变更，发现盲点和潜在问题。

### 测试与交互

- 需要与 RoboClaw agent 交互时，始终使用 `roboclaw agent --logs` 以获取完整的运行时信息。

### 产品原则

- RoboClaw 对用户的对话必须保持高层次和通用化。不暴露串口路径、底层协议细节、内部技术实现。用户应无感地完成操作。

### 提交前审查

- commit 前必须用 `/simplify` 进行 review，检查代码规范、复用、质量和效率。发现问题先修再提交。
- 当用户要求多角度审查，或改动超过 300 行时，额外用 `/codex-review` 从独立视角审查。
- 审查必须覆盖本文件中的所有代码规范（缩进层数、文件行数、try/except、复用等），不能只看功能正确性。

### 代码规范

- 框架代码放 `roboclaw/embodied/`，用户资产放 `~/.roboclaw/workspace/embodied/`。
- 不写向后兼容代码。不保留旧接口、不做 fallback 适配、不写 deprecated wrapper。旧的不要了就直接删掉。
- 不用 try/except 吞错误。有报错就让它直接抛出来，不要静默捕获。只在确实需要处理特定异常时才 catch。
- 复用优先。已有实现的功能不要重复造轮子，先找现有代码再决定是否新写。
- 单个 .py 文件不超过 1000 行。超过时必须将独立逻辑拆分到单独的模块。
- 嵌套不超过 3 层缩进。超过时应将内层逻辑提取为独立函数。

---
