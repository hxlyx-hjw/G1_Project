"""
================================================================
对话历史离线存储（SQLite）
================================================================

依赖：无（使用 Python 内置的 sqlite3，完全离线，无需服务器）

数据库文件默认是 chat_history.db，放在当前目录。
包含两张表：
    conversations —— 会话表（每次对话一个会话）
    messages      —— 消息表（每条用户/助手消息一行）

用法：
    from chat_store import ChatStore

    store = ChatStore("chat_history.db")

    cid = store.create_conversation("关于天气的讨论")
    store.add_message(cid, "user", "今天天气怎么样？")
    store.add_message(cid, "assistant", "晴天，20 度。")

    print(store.get_messages(cid))        # 查某次会话的所有消息
    print(store.list_conversations())     # 列出所有会话

    store.close()                          # 用完后关闭连接
"""

import sqlite3
from datetime import datetime
from typing import List, Optional


class ChatStore:
    """基于 SQLite 的对话历史存储，线程安全、零依赖。"""

    def __init__(self, db_path: str = "chat_history.db"):
        # check_same_thread=False 允许多线程环境使用
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row   # 让查询结果可以用列名访问
        self._init_tables()

    # ------------------------------------------------------------
    # 建表
    # ------------------------------------------------------------
    def _init_tables(self) -> None:
        """创建会话表和消息表（如果不存在）。"""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT NOT NULL,          -- 会话标题
                created_at TEXT NOT NULL,          -- 创建时间
                updated_at TEXT NOT NULL           -- 最后更新时间
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,  -- 所属会话
                role            TEXT NOT NULL,     -- user / assistant / system / tool
                content         TEXT NOT NULL,     -- 消息内容
                created_at      TEXT NOT NULL,     -- 消息时间
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------
    @staticmethod
    def _now() -> str:
        """返回当前时间字符串，格式：YYYY-MM-DD HH:MM:SS。"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------
    # 会话相关
    # ------------------------------------------------------------
    def create_conversation(self, title: str) -> int:
        """新建一个会话，返回会话 id。"""
        now = self._now()
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        self.conn.commit()
        return cur.lastrowid

    def list_conversations(self, limit: int = 50) -> List[dict]:
        """按最后更新时间倒序，列出最近的会话。"""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, title, created_at, updated_at "
            "FROM conversations ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return [
            {"id": r["id"], "title": r["title"],
             "created_at": r["created_at"], "updated_at": r["updated_at"]}
            for r in cur.fetchall()
        ]

    def delete_conversation(self, conversation_id: int) -> None:
        """删除某个会话及其所有消息。"""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cur.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self.conn.commit()

    # ------------------------------------------------------------
    # 消息相关
    # ------------------------------------------------------------
    def add_message(self, conversation_id: int, role: str, content: str) -> None:
        """往会话里追加一条消息，并刷新会话的更新时间。"""
        now = self._now()
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        cur.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        self.conn.commit()

    def get_messages(self, conversation_id: int) -> List[dict]:
        """按时间顺序返回某会话的所有消息。"""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT role, content, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        )
        return [
            {"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
            for r in cur.fetchall()
        ]

    def search_messages(self, keyword: str, limit: int = 50) -> List[dict]:
        """按关键词搜索所有消息内容（LIKE 模糊匹配），返回命中的消息。

        每条结果带所属会话 id 和标题，方便定位来源。
        """
        cur = self.conn.cursor()
        cur.execute(
            "SELECT m.role, m.content, m.created_at, c.id AS conversation_id, c.title "
            "FROM messages m JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.content LIKE ? ORDER BY m.id DESC LIMIT ?",
            (f"%{keyword}%", limit),
        )
        return [
            {
                "role": r["role"], "content": r["content"], "created_at": r["created_at"],
                "conversation_id": r["conversation_id"], "title": r["title"],
            }
            for r in cur.fetchall()
        ]

    def export_conversation(self, conversation_id: int, fmt: str = "md") -> str:
        """把某会话导出为文本字符串（fmt 支持 md / txt）。"""
        msgs = self.get_messages(conversation_id)
        if not msgs:
            return "（该会话没有消息）"

        labels = {"user": "用户", "assistant": "助手", "system": "系统", "tool": "工具"}
        lines = []
        for m in msgs:
            label = labels.get(m["role"], m["role"])
            if fmt == "md":
                lines.append(f"## {label}\n\n{m['content']}\n")
            else:  # txt
                lines.append(f"{label}：{m['content']}")

        return "\n".join(lines)

    # ------------------------------------------------------------
    # 关闭
    # ------------------------------------------------------------
    def close(self) -> None:
        """关闭数据库连接。"""
        self.conn.close()


# ============================================================
# 命令行入口：方便直接查看数据库内容
# ============================================================
if __name__ == "__main__":
    import sys

    store = ChatStore()
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        for c in store.list_conversations():
            print(f"[{c['id']}] {c['title']}  更新于 {c['updated_at']}")
    elif len(sys.argv) > 1 and sys.argv[1].isdigit():
        for m in store.get_messages(int(sys.argv[1])):
            print(f"{m['role']}: {m['content']}")
    else:
        print("用法：")
        print("  python chat_store.py list        # 列出所有会话")
        print("  python chat_store.py <会话id>     # 查看某会话内容")
    store.close()
