# 🔒 安全指南

## ⚠️ API Key 安全警告

如果 `config/config.yaml` 文件已包含真实的 API Key 并被提交到 Git 仓库，请**立即**执行以下步骤：

### 1. 从 Git 历史中移除敏感信息（如果已提交）

```bash
# 从 Git 跟踪中移除（保留本地文件）
git rm --cached config/config.yaml

# 如果已经提交到了远程仓库，需要重写历史
# ⚠️ 警告：这会重写 Git 历史，影响所有协作者
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch config/config.yaml" \
  --prune-empty --tag-name-filter cat -- --all

# 强制推送到远程（谨慎使用）
# git push origin --force --all
```

### 2. 撤销并重新生成 API Key

**这是最重要的步骤！**

1. 登录你的 API 提供商（硅基流动/Qwen等）
2. **立即撤销**已泄露的 API Key
3. 生成新的 API Key
4. 更新本地 `config/config.yaml` 文件

### 3. 确保不再提交敏感信息

✅ **已完成**：`config/config.yaml` 已添加到 `.gitignore`

验证配置：
```bash
# 检查 .gitignore 是否包含 config.yaml
grep "config.yaml" .gitignore

# 检查 Git 状态，确认 config.yaml 不再被跟踪
git status config/config.yaml
```

### 4. 使用示例配置文件

✅ **已创建**：`config/config.yaml.example` 作为模板

首次设置：
```bash
# 复制示例文件
cp config/config.yaml.example config/config.yaml

# 编辑并填入你的 API Key
# （使用你喜欢的编辑器，如 vim, nano, VS Code 等）
```

## 📋 安全检查清单

- [ ] 从 Git 中移除了包含 API Key 的 `config.yaml`
- [ ] 撤销了已泄露的 API Key
- [ ] 生成了新的 API Key
- [ ] 确认 `.gitignore` 包含 `config/config.yaml`
- [ ] 确认 Git 不再跟踪 `config.yaml`
- [ ] 使用 `config.yaml.example` 作为新配置的模板

## 🔐 最佳实践

1. **永远不要**将包含 API Key 的文件提交到 Git
2. **始终使用** `.gitignore` 保护敏感配置文件
3. **定期轮换** API Key（建议每3-6个月）
4. **限制权限**：仅授予 API Key 必要的最小权限
5. **监控使用**：定期检查 API 使用情况，发现异常立即撤销

## 📚 相关资源

- Git 安全最佳实践：https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History
- API Key 管理指南：参考你的 API 提供商文档

---

**安全第一！** 如果发现 API Key 泄露，立即撤销并重新生成。

