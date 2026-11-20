"""
================================================================
文件内容解析封装：支持 txt / md / pdf / word(.docx)
================================================================

依赖安装：
    pip install pypdf python-docx

用法（自动按扩展名分发）：
    from file_parser import parse_file
    text = parse_file(r"C:\docs\报告.pdf")
    print(text)

单独调用某个解析器：
    from file_parser import parse_pdf, parse_docx
    print(parse_docx(r"C:\docs\合同.docx"))

命令行直接读文件：
    python file_parser.py C:\docs\报告.pdf
"""

from pathlib import Path


# ============================================================
# 1. 各类型文件的解析函数
# ============================================================

def parse_txt(path) -> str:
    """解析纯文本文件（.txt / .log / .csv / .json 等），按 UTF-8 读取。"""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def parse_md(path) -> str:
    """解析 Markdown 文件（本质是纯文本，直接读原始内容即可）。"""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def parse_pdf(path) -> str:
    """解析 PDF，逐页提取文本并合并。

    优先用新版库 pypdf，旧项目里可能是 PyPDF2，做了兼容导入。
    """
    try:
        from pypdf import PdfReader       # 新版库名
    except ImportError:
        from PyPDF2 import PdfReader      # 兼容旧版库名

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)             # 页与页之间空行分隔


def parse_docx(path) -> str:
    """解析 Word (.docx)，提取段落文本和表格内容。"""
    from docx import Document

    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]

    # 表格里的文字也一并提取，避免漏掉
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            parts.append(" | ".join(cells))

    return "\n".join(parts)


# ============================================================
# 2. 扩展名 → 解析器 映射表
# ============================================================
PARSERS = {
    ".txt":      parse_txt,
    ".log":      parse_txt,
    ".csv":      parse_txt,
    ".json":     parse_txt,
    ".py":       parse_txt,
    ".html":     parse_txt,
    ".xml":      parse_txt,
    ".md":       parse_md,
    ".markdown": parse_md,
    ".pdf":      parse_pdf,
    ".docx":     parse_docx,
    # 注意：旧版 .doc（二进制格式）python-docx 无法解析，
    # 需要先用 Word / LibreOffice 另存为 .docx
}


# ============================================================
# 3. 统一入口：按扩展名自动分发
# ============================================================
def parse_file(path) -> str:
    """根据文件扩展名自动选择解析器，返回文本内容。

    设计成「永不抛异常」：文件不存在、类型不支持、解析出错时，
    都返回一段带说明的文字，方便直接喂给大模型，不会让调用方崩溃。
    """
    p = Path(path)
    if not p.exists():
        return f"文件不存在：{path}"

    ext = p.suffix.lower()
    parser = PARSERS.get(ext)
    if parser is None:
        supported = "、".join(sorted(PARSERS))
        return f"不支持的文件类型：{ext}（当前支持：{supported}）"

    try:
        return parser(p)
    except Exception as exc:  # noqa: BLE001
        return f"解析失败 {p.name}：{exc}"


# ============================================================
# 4. 命令行入口
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print(parse_file(sys.argv[1]))
    else:
        print("用法：python file_parser.py <文件路径>")
