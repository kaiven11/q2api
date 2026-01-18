#!/usr/bin/env python3
"""
Q2API 账户状态监控脚本
定期检测账户状态，自动禁用有问题的账户，避免影响正常用户使用
"""

import sys
import asyncio
import json
import time
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from parent directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Add parent directory to path for imports
sys.path.insert(0, str(BASE_DIR))

from db import init_db, close_db, row_to_dict

# 配置参数
CONFIG = {
    # 连续失败多少次后禁用账户
    'max_consecutive_failures': 3,
    # 错误率超过多少后禁用账户 (0.8 = 80%)
    'max_error_rate': 0.8,
    # 最小调用次数，低于此数不计算错误率
    'min_calls_for_error_rate': 10,
    # Token刷新失败多少小时后禁用账户
    'refresh_failure_hours': 6,
    # 是否自动重新启用恢复的账户
    'auto_reenable': True,
    # 重新启用前的等待时间（小时）
    'reenable_wait_hours': 2,
    # 日志文件路径
    'log_file': BASE_DIR / 'logs' / 'account_monitor.log'
}

def check_docker_logs_for_errors():
    """检查Docker日志中的403错误"""
    try:
        # 获取最近10分钟的Docker日志
        result = subprocess.run([
            'docker', 'logs', '--since', '10m', 'q2api'
        ], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            log_message(f"获取Docker日志失败: {result.stderr}", "ERROR")
            return {}

        logs = result.stdout
        error_accounts = {}

        # 匹配403错误的账户
        pattern = r'\[403 Error\] Account ([a-f0-9-]+) \(([^)]+)\) got 403'
        matches = re.findall(pattern, logs)

        for account_id, label in matches:
            if account_id not in error_accounts:
                error_accounts[account_id] = {
                    'label': label,
                    'error_count': 0,
                    'last_error': None
                }
            error_accounts[account_id]['error_count'] += 1

        # 匹配刷新失败
        refresh_pattern = r'\[403 Failed\] Account ([a-f0-9-]+) retry failed'
        refresh_matches = re.findall(refresh_pattern, logs)

        for account_id in refresh_matches:
            if account_id in error_accounts:
                error_accounts[account_id]['refresh_failed'] = True

        return error_accounts

    except subprocess.TimeoutExpired:
        log_message("Docker日志检查超时", "ERROR")
        return {}
    except Exception as e:
        log_message(f"检查Docker日志时出错: {e}", "ERROR")
        return {}
    """记录日志消息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"

    # 打印到控制台
    print(log_entry)

    # 写入日志文件
    try:
        CONFIG['log_file'].parent.mkdir(exist_ok=True)
        with open(CONFIG['log_file'], 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    except Exception as e:
        print(f"写入日志文件失败: {e}")

async def check_account_health(account: dict) -> dict:
    """检查单个账户的健康状态"""
    account_id = account.get('id')
    label = account.get('label', f'Account-{account_id}')

    issues = []
    recommendations = []

    # 检查1: Token刷新状态
    last_refresh_status = account.get('last_refresh_status')
    last_refresh_time = account.get('last_refresh_time')

    if last_refresh_status == 'failed':
        if last_refresh_time:
            try:
                refresh_time = datetime.fromisoformat(last_refresh_time.replace('Z', '+00:00'))
                hours_since_failure = (datetime.now() - refresh_time).total_seconds() / 3600

                if hours_since_failure > CONFIG['refresh_failure_hours']:
                    issues.append(f"Token刷新失败超过{CONFIG['refresh_failure_hours']}小时")
                    recommendations.append("disable")
            except Exception:
                issues.append("Token刷新失败且时间解析错误")
                recommendations.append("disable")

    # 检查2: 错误率
    success_count = account.get('success_count', 0)
    error_count = account.get('error_count', 0)
    total_calls = success_count + error_count

    if total_calls >= CONFIG['min_calls_for_error_rate']:
        error_rate = error_count / total_calls
        if error_rate > CONFIG['max_error_rate']:
            issues.append(f"错误率过高: {error_rate:.2%} (成功:{success_count}, 错误:{error_count})")
            recommendations.append("disable")

    # 检查3: 连续失败（这里简化处理，基于error_count和success_count的比例）
    if error_count > 0 and success_count == 0 and error_count >= CONFIG['max_consecutive_failures']:
        issues.append(f"连续失败{error_count}次，无成功记录")
        recommendations.append("disable")

    # 检查4: 账户是否应该重新启用
    if not account.get('enabled') and CONFIG['auto_reenable']:
        # 如果账户被禁用，检查是否可以重新启用
        if last_refresh_status != 'failed' and error_count == 0:
            recommendations.append("reenable")
            issues.append("账户状态已恢复，建议重新启用")

    return {
        'account_id': account_id,
        'label': label,
        'issues': issues,
        'recommendations': recommendations,
        'current_status': 'enabled' if account.get('enabled') else 'disabled'
    }

async def apply_recommendations(db, account_id: int, recommendations: list, label: str):
    """应用推荐的操作"""
    for action in recommendations:
        if action == "disable":
            try:
                await db.execute(
                    "UPDATE accounts SET enabled = 0 WHERE id = ?",
                    (account_id,)
                )
                log_message(f"已禁用账户 {label} (ID: {account_id})", "ACTION")
            except Exception as e:
                log_message(f"禁用账户 {label} 失败: {e}", "ERROR")

        elif action == "reenable":
            try:
                await db.execute(
                    "UPDATE accounts SET enabled = 1 WHERE id = ?",
                    (account_id,)
                )
                log_message(f"已重新启用账户 {label} (ID: {account_id})", "ACTION")
            except Exception as e:
                log_message(f"重新启用账户 {label} 失败: {e}", "ERROR")

async def monitor_accounts():
    """主监控函数"""
    log_message("开始账户状态监控")

    db = await init_db()

    try:
        # 获取所有账户
        accounts = await db.fetchall("SELECT * FROM accounts ORDER BY id")
        accounts = [row_to_dict(acc) for acc in accounts]

        if not accounts:
            log_message("未找到任何账户")
            return

        log_message(f"检查 {len(accounts)} 个账户")

        total_issues = 0
        actions_taken = 0

        for account in accounts:
            health_check = await check_account_health(account)

            if health_check['issues']:
                total_issues += len(health_check['issues'])
                log_message(
                    f"账户 {health_check['label']} (ID: {health_check['account_id']}) "
                    f"发现问题: {'; '.join(health_check['issues'])}",
                    "WARN"
                )

                # 应用推荐操作
                if health_check['recommendations']:
                    await apply_recommendations(
                        db,
                        health_check['account_id'],
                        health_check['recommendations'],
                        health_check['label']
                    )
                    actions_taken += len(health_check['recommendations'])

        # 统计信息
        enabled_count = sum(1 for acc in accounts if acc.get('enabled'))
        disabled_count = len(accounts) - enabled_count

        log_message(
            f"监控完成 - 总账户: {len(accounts)}, "
            f"启用: {enabled_count}, 禁用: {disabled_count}, "
            f"发现问题: {total_issues}, 执行操作: {actions_taken}"
        )

    except Exception as e:
        log_message(f"监控过程中发生错误: {e}", "ERROR")
        import traceback
        log_message(f"错误详情: {traceback.format_exc()}", "ERROR")

    finally:
        await close_db()

async def test_account_connection():
    """测试账户连接（可选的更深度检查）"""
    # 这里可以添加实际的API调用测试
    # 但为了不影响正常使用，建议谨慎使用
    pass

def main():
    """脚本主入口"""
    try:
        asyncio.run(monitor_accounts())
    except KeyboardInterrupt:
        log_message("监控被用户中断", "INFO")
    except Exception as e:
        log_message(f"脚本执行失败: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()