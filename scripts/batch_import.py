#!/usr/bin/env python3
"""
Q2API 批量导入 Amazon Q Developer 账户脚本
支持从格式化的账户数据批量导入到数据库
"""

import sys
import asyncio
import uuid
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from parent directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Add parent directory to path for imports
sys.path.insert(0, str(BASE_DIR))

from db import init_db, close_db

def parse_account_line(line: str) -> dict:
    """解析单行账户数据"""
    try:
        parts = line.strip().split('|')
        if len(parts) < 6:
            raise ValueError(f"Invalid format: expected at least 6 parts, got {len(parts)}")

        email = parts[0]
        password = parts[1]
        client_id = parts[2]
        client_secret = parts[3]
        refresh_token = parts[4]
        access_token = parts[5]

        # 生成唯一ID
        account_id = str(uuid.uuid4())

        # 从email提取标签
        label = email.split('@')[0] if '@' in email else email

        return {
            'id': account_id,
            'label': label,
            'clientId': client_id,
            'clientSecret': client_secret,
            'refreshToken': refresh_token,
            'accessToken': access_token,
            'other': json.dumps({
                'email': email,
                'password': password,
                'imported_at': datetime.now().isoformat()
            }),
            'last_refresh_time': datetime.now().isoformat(),
            'last_refresh_status': 'success',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'enabled': 1,
            'error_count': 0,
            'success_count': 0
        }
    except Exception as e:
        raise ValueError(f"Failed to parse line: {e}")

async def import_accounts(account_data: str, dry_run: bool = False):
    """批量导入账户"""
    lines = [line.strip() for line in account_data.strip().split('\n') if line.strip()]

    if not lines:
        print("❌ 没有找到有效的账户数据")
        return

    print(f"📊 准备导入 {len(lines)} 个账户")

    accounts = []
    errors = []

    # 解析所有账户
    for i, line in enumerate(lines, 1):
        try:
            account = parse_account_line(line)
            accounts.append(account)
            print(f"✅ 账户 {i}: {account['label']} ({account['id'][:8]}...)")
        except Exception as e:
            error_msg = f"❌ 账户 {i}: {str(e)}"
            errors.append(error_msg)
            print(error_msg)

    if errors:
        print(f"\n⚠️  发现 {len(errors)} 个解析错误:")
        for error in errors:
            print(f"  {error}")

    if not accounts:
        print("❌ 没有成功解析的账户，退出")
        return

    if dry_run:
        print(f"\n🔍 干跑模式: 将导入 {len(accounts)} 个账户")
        for account in accounts:
            print(f"  - {account['label']} (ID: {account['id'][:8]}...)")
        return

    # 连接数据库并导入
    db = await init_db()

    try:
        imported_count = 0
        duplicate_count = 0

        for account in accounts:
            try:
                # 检查是否已存在相同的clientId
                existing = await db.fetchone(
                    "SELECT id FROM accounts WHERE clientId = ?",
                    (account['clientId'],)
                )

                if existing:
                    print(f"⚠️  跳过重复账户: {account['label']} (clientId已存在)")
                    duplicate_count += 1
                    continue

                # 插入新账户
                await db.execute("""
                    INSERT INTO accounts (
                        id, label, clientId, clientSecret, refreshToken, accessToken,
                        other, last_refresh_time, last_refresh_status, created_at,
                        updated_at, enabled, error_count, success_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    account['id'], account['label'], account['clientId'],
                    account['clientSecret'], account['refreshToken'], account['accessToken'],
                    account['other'], account['last_refresh_time'], account['last_refresh_status'],
                    account['created_at'], account['updated_at'], account['enabled'],
                    account['error_count'], account['success_count']
                ))

                imported_count += 1
                print(f"✅ 导入成功: {account['label']}")

            except Exception as e:
                print(f"❌ 导入失败 {account['label']}: {e}")

        print(f"\n🎉 导入完成!")
        print(f"  - 成功导入: {imported_count} 个账户")
        print(f"  - 跳过重复: {duplicate_count} 个账户")
        print(f"  - 解析错误: {len(errors)} 个账户")

    except Exception as e:
        print(f"❌ 数据库操作失败: {e}")
    finally:
        await close_db()

def main():
    """脚本主入口"""
    if len(sys.argv) < 2:
        print("用法:")
        print("  python3 batch_import.py <账户数据文件> [--dry-run]")
        print("  python3 batch_import.py - [--dry-run]  # 从标准输入读取")
        print("")
        print("账户数据格式 (每行一个账户):")
        print("  email|password|clientId|clientSecret|refreshToken|accessToken")
        sys.exit(1)

    file_path = sys.argv[1]
    dry_run = '--dry-run' in sys.argv

    try:
        if file_path == '-':
            print("📝 请粘贴账户数据 (Ctrl+D 结束输入):")
            account_data = sys.stdin.read()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                account_data = f.read()

        if dry_run:
            print("🔍 干跑模式: 只解析不导入")

        asyncio.run(import_accounts(account_data, dry_run))

    except FileNotFoundError:
        print(f"❌ 文件不存在: {file_path}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()