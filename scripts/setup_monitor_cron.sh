#!/bin/bash

# Q2API 账户监控 Crontab 设置脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Q2API_DIR="$(dirname "$SCRIPT_DIR")"
MONITOR_SCRIPT="$Q2API_DIR/scripts/account_monitor.py"
LOG_DIR="$Q2API_DIR/logs"

echo "Q2API 账户监控 Crontab 设置"
echo "================================"
echo "脚本目录: $SCRIPT_DIR"
echo "Q2API目录: $Q2API_DIR"
echo "监控脚本: $MONITOR_SCRIPT"

# 检查监控脚本是否存在
if [ ! -f "$MONITOR_SCRIPT" ]; then
    echo "错误: 监控脚本不存在: $MONITOR_SCRIPT"
    exit 1
fi

# 创建日志目录
mkdir -p "$LOG_DIR"

# 使监控脚本可执行
chmod +x "$MONITOR_SCRIPT"

echo ""
echo "推荐的监控间隔设置："
echo "1. 每2小时 (推荐) - 平衡监控效果和系统负载"
echo "2. 每1小时 - 更频繁监控，适合高负载环境"
echo "3. 每4小时 - 较少监控，适合稳定环境"
echo "4. 每30分钟 - 高频监控，仅在问题频发时使用"

echo ""
read -p "请选择监控间隔 (1-4): " choice

case $choice in
    1)
        CRON_SCHEDULE="0 */2 * * *"
        DESCRIPTION="每2小时"
        ;;
    2)
        CRON_SCHEDULE="0 * * * *"
        DESCRIPTION="每1小时"
        ;;
    3)
        CRON_SCHEDULE="0 */4 * * *"
        DESCRIPTION="每4小时"
        ;;
    4)
        CRON_SCHEDULE="*/30 * * * *"
        DESCRIPTION="每30分钟"
        ;;
    *)
        echo "无效选择，使用默认设置：每2小时"
        CRON_SCHEDULE="0 */2 * * *"
        DESCRIPTION="每2小时"
        ;;
esac

# 生成crontab条目
CRON_ENTRY="$CRON_SCHEDULE cd $Q2API_DIR && /usr/bin/python3 $MONITOR_SCRIPT >> $LOG_DIR/cron_monitor.log 2>&1"

echo ""
echo "将添加以下crontab条目："
echo "$CRON_ENTRY"
echo ""
echo "说明: $DESCRIPTION执行一次账户状态检查"

read -p "确认添加到crontab? (y/N): " confirm

if [[ $confirm =~ ^[Yy]$ ]]; then
    # 备份现有crontab
    crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true

    # 检查是否已存在相同的监控任务
    if crontab -l 2>/dev/null | grep -q "account_monitor.py"; then
        echo "警告: 检测到已存在的账户监控任务"
        read -p "是否替换现有任务? (y/N): " replace

        if [[ $replace =~ ^[Yy]$ ]]; then
            # 移除现有的监控任务
            crontab -l 2>/dev/null | grep -v "account_monitor.py" | crontab -
            echo "已移除现有监控任务"
        else
            echo "取消操作"
            exit 0
        fi
    fi

    # 添加新的监控任务
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

    echo "✅ Crontab任务添加成功!"
    echo ""
    echo "监控配置："
    echo "- 执行频率: $DESCRIPTION"
    echo "- 脚本路径: $MONITOR_SCRIPT"
    echo "- 日志文件: $LOG_DIR/account_monitor.log"
    echo "- Cron日志: $LOG_DIR/cron_monitor.log"
    echo ""
    echo "查看当前crontab: crontab -l"
    echo "查看监控日志: tail -f $LOG_DIR/account_monitor.log"
    echo "手动执行监控: cd $Q2API_DIR && python3 $MONITOR_SCRIPT"

    # 立即执行一次测试
    read -p "是否立即执行一次监控测试? (y/N): " test_run
    if [[ $test_run =~ ^[Yy]$ ]]; then
        echo ""
        echo "执行监控测试..."
        cd "$Q2API_DIR"
        python3 "$MONITOR_SCRIPT"
        echo ""
        echo "测试完成，请检查日志文件确认正常运行"
    fi

else
    echo "操作已取消"
fi

echo ""
echo "监控策略说明："
echo "- 自动禁用: Token刷新失败超过6小时的账户"
echo "- 自动禁用: 错误率超过80%且调用次数≥10的账户"
echo "- 自动禁用: 连续失败3次以上的账户"
echo "- 自动重启: 状态恢复的账户会被重新启用"
echo "- 日志记录: 所有操作都会记录到日志文件"