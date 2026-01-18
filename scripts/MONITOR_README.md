# Q2API 账户监控系统

## 概述

这个监控系统会定期检查 Q2API 中的 Claude 账户状态，自动禁用有问题的账户，避免影响正常用户使用。

## 文件说明

- `account_monitor.py` - 主监控脚本
- `setup_monitor_cron.sh` - Crontab 设置脚本
- `logs/account_monitor.log` - 监控日志
- `logs/cron_monitor.log` - Cron 执行日志

## 快速开始

### 1. 设置定时任务

```bash
cd /root/q2api/scripts
chmod +x setup_monitor_cron.sh account_monitor.py
./setup_monitor_cron.sh
```

### 2. 手动执行监控

```bash
cd /root/q2api
python3 scripts/account_monitor.py
```

### 3. 查看日志

```bash
# 查看监控日志
tail -f /root/q2api/logs/account_monitor.log

# 查看 Cron 执行日志
tail -f /root/q2api/logs/cron_monitor.log
```

## 监控策略

### 自动禁用条件

1. **Token 刷新失败**: 超过 6 小时未成功刷新
2. **高错误率**: 错误率超过 80% 且调用次数 ≥ 10
3. **连续失败**: 连续失败 3 次以上且无成功记录

### 自动重启条件

- 账户状态恢复正常（Token 刷新成功且无错误）

## 推荐的监控间隔

| 间隔 | 适用场景 | 说明 |
|------|----------|------|
| **2小时** | 推荐设置 | 平衡监控效果和系统负载 |
| 1小时 | 高负载环境 | 更频繁监控，快速响应问题 |
| 4小时 | 稳定环境 | 较少监控，适合问题较少的环境 |
| 30分钟 | 紧急情况 | 高频监控，仅在问题频发时使用 |

## 配置参数

在 `account_monitor.py` 中可以调整以下参数：

```python
CONFIG = {
    'max_consecutive_failures': 3,      # 连续失败次数阈值
    'max_error_rate': 0.8,             # 最大错误率 (80%)
    'min_calls_for_error_rate': 10,    # 计算错误率的最小调用次数
    'refresh_failure_hours': 6,        # Token刷新失败容忍时间
    'auto_reenable': True,             # 是否自动重新启用
    'reenable_wait_hours': 2,          # 重新启用等待时间
}
```

## 常用命令

```bash
# 查看当前 Crontab
crontab -l

# 编辑 Crontab
crontab -e

# 移除监控任务
crontab -l | grep -v "account_monitor.py" | crontab -

# 查看账户统计
cd /root/q2api && python3 scripts/account_stats.py

# 实时监控日志
tail -f /root/q2api/logs/account_monitor.log
```

## 日志格式

```
[2026-01-10 14:30:00] [INFO] 开始账户状态监控
[2026-01-10 14:30:01] [WARN] 账户 Account-1 (ID: 1) 发现问题: Token刷新失败超过6小时
[2026-01-10 14:30:01] [ACTION] 已禁用账户 Account-1 (ID: 1)
[2026-01-10 14:30:02] [INFO] 监控完成 - 总账户: 5, 启用: 4, 禁用: 1, 发现问题: 1, 执行操作: 1
```

## 故障排除

### 1. 脚本执行失败

```bash
# 检查 Python 环境
python3 --version

# 检查依赖
cd /root/q2api && python3 -c "import asyncio, sqlite3; print('依赖正常')"

# 检查数据库
cd /root/q2api && python3 scripts/account_stats.py
```

### 2. Crontab 不执行

```bash
# 检查 Cron 服务
systemctl status cron

# 查看 Cron 日志
tail -f /var/log/cron

# 检查脚本权限
ls -la /root/q2api/scripts/account_monitor.py
```

### 3. 权限问题

```bash
# 设置正确权限
chmod +x /root/q2api/scripts/account_monitor.py
chown root:root /root/q2api/scripts/account_monitor.py
```

## 安全注意事项

1. **谨慎禁用**: 监控脚本会自动禁用账户，请确保配置合理
2. **日志监控**: 定期检查日志，确保监控正常运行
3. **备份策略**: 建议在重要操作前备份数据库
4. **测试环境**: 在生产环境使用前，先在测试环境验证

## 监控效果

- **减少 403 错误**: 及时禁用有问题的账户
- **提高稳定性**: 避免使用失效的账户影响服务
- **自动恢复**: 账户恢复后自动重新启用
- **详细日志**: 完整记录所有监控和操作行为