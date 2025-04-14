import os
import re
import unicodedata

# Đường dẫn tuyệt đối tới thư mục chứa utils.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn đầy đủ tới stopword file
STOP_WORD_FILE = os.path.join(BASE_DIR, 'Data', 'vietnamese-stopwords.txt')

# Đọc stop words vào set để tra cứu nhanh
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = set(word.strip() for word in file if word.strip())

# Hàm kiểm tra từ có phải là từ tiếng Việt "sạch"
def is_valid_vietnamese(word):
    vietnamese_chars = (
        "a-zA-Z0-9_"
        "àáạảãâầấậẩẫăằắặẳẵ"
        "èéẹẻẽêềếệểễ"
        "ìíịỉĩ"
        "òóọỏõôồốộổỗơờớợởỡ"
        "ùúụủũưừứựửữ"
        "ỳýỵỷỹ"
        "đ"
        "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ"
        "ÈÉẸẺẼÊỀẾỆỂỄ"
        "ÌÍỊỈĨ"
        "ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
        "ÙÚỤỦŨƯỪỨỰỬỮ"
        "ỲÝỴỶỸ"
        "Đ"
    )
    pattern = f'^[{vietnamese_chars}]+$'
    return re.match(pattern, word) is not None

# Hàm lọc từ tiếng Việt hợp lệ
def filter_vietnamese_words(text):
    if not isinstance(text, str):
        return ''
    words = text.split()
    clean_words = [w for w in words if is_valid_vietnamese(w)]
    return ' '.join(clean_words)

# Hàm làm sạch văn bản
def clean_text_row(text, stop_words):
    # 1️⃣ Xóa số
    text = [re.sub(r'\d+', '', t) for t in text]

    # 2️⃣ Loại bỏ ký tự đặc biệt, giữ lại chữ cái & dấu cách
    text = [re.sub(r'[^a-zA-ZÀ-Ỹà-ỹ0-9\s_]', '', t) for t in text]

    # 3️⃣ Chuẩn hóa Unicode
    text = [unicodedata.normalize("NFKC", t) for t in text]

    # 4️⃣ Chuyển về chữ thường
    text = [t.lower() for t in text]

    # 5️⃣ Loại bỏ từ quá ngắn
    text = [t for t in text if len(t) > 1]

    # 6️⃣ Loại bỏ stop words
    text = [t for t in text if t not in stop_words]

    return text
