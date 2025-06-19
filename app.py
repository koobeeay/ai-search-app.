# app.py

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import fitz  # PyMuPDF
import os

# --- 1. CẤU HÌNH VÀ KHỞI TẠO BAN ĐẦU ---

# Thiết lập layout cho trang web
st.set_page_config(layout="wide", page_title="Trợ lý Tìm kiếm Thông minh")

# Định nghĩa tên file để lưu trữ
DATA_FILE = "data.json"
INDEX_FILE = "index.faiss"

# Tải mô hình AI (sử dụng cache để tối ưu)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- 2. CÁC HÀM XỬ LÝ DỮ LIỆU (LƯU, TẢI, CẬP NHẬT) ---

def save_data(texts, index):
    """Lưu văn bản và chỉ mục FAISS vào file."""
    if texts:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False, indent=4)
    if index is not None:
        faiss.write_index(index, INDEX_FILE)

def load_data():
    """Tải dữ liệu từ file khi khởi động."""
    texts = []
    index = None
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            texts = json.load(f)
    if os.path.exists(INDEX_FILE) and texts:
        index = faiss.read_index(INDEX_FILE)
    return texts, index

def update_index(texts):
    """Tạo hoặc cập nhật chỉ mục FAISS từ danh sách văn bản."""
    if not texts:
        return None
    st.session_state.is_processing = True
    with st.spinner("🧠 AI đang phân tích và lập chỉ mục cho tài liệu..."):
        text_embeddings = model.encode(texts)
        d = text_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(text_embeddings)
    st.session_state.is_processing = False
    return index

# --- 3. KHỞI TẠO SESSION STATE ---

# Tải dữ liệu lần đầu tiên
if 'texts' not in st.session_state:
    st.session_state.texts, st.session_state.index = load_data()

if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

# --- 4. GIAO DIỆN CHÍNH ---

st.title("👨‍💻 Trợ lý Tìm kiếm Tài liệu Thông minh")
st.write("Sử dụng AI để tìm kiếm chính xác nội dung trong kho tài liệu của bạn.")

# Khu vực tìm kiếm
query = st.text_input(
    "👇 Đặt câu hỏi hoặc nhập từ khóa cần tìm vào đây:",
    placeholder="Ví dụ: Báo cáo doanh thu quý 4 năm 2024 có những điểm gì nổi bật?"
)

if query and st.session_state.index:
    k = st.slider("Số lượng kết quả phù hợp nhất:", 1, 10, 3)
    with st.spinner("🔍 Đang tìm kiếm..."):
        query_embedding = model.encode([query])
        distances, indices = st.session_state.index.search(query_embedding, k)
    
    st.header("💡 Kết quả tìm kiếm:")
    for i in range(len(indices[0])):
        text_index = indices[0][i]
        if 0 <= text_index < len(st.session_state.texts):
            relevance_score = max(0, 1 - distances[0][i])
            st.write(f"#### **Kết quả {i+1}** (Độ liên quan: {relevance_score:.2%})")
            st.success(st.session_state.texts[text_index])
elif query:
    st.warning("⚠️ Kho dữ liệu đang trống. Vui lòng thêm tài liệu ở thanh bên trái để bắt đầu.")

# --- 5. THANH BÊN (SIDEBAR) ---

with st.sidebar:
    st.header("🗂️ Quản lý Kho dữ liệu")
    
    # Chức năng tải file lên
    uploaded_files = st.file_uploader(
        "Thêm tài liệu mới (.txt, .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )
    
    if st.button("Thêm vào kho", use_container_width=True, type="primary", disabled=st.session_state.is_processing):
        if uploaded_files:
            new_texts = []
            for file in uploaded_files:
                content = ""
                if file.type == "text/plain":
                    content = file.read().decode("utf-8")
                elif file.type == "application/pdf":
                    with fitz.open(stream=file.read(), filetype="pdf") as doc:
                        content = "".join(page.get_text() for page in doc)
                if content:
                    new_texts.append(f"--- Nguồn: {file.name} ---\n{content}")
            
            st.session_state.texts.extend(new_texts)
            st.session_state.index = update_index(st.session_state.texts)
            save_data(st.session_state.texts, st.session_state.index)
            st.success(f"Đã thêm {len(uploaded_files)} file!")
            st.rerun() # Tải lại trang để cập nhật
        else:
            st.warning("Vui lòng chọn file để thêm.")
            
    # Hiển thị và quản lý các tài liệu hiện có
    if st.session_state.texts:
        st.write("---")
        st.subheader("Tài liệu trong kho")
        
        with st.expander(f"Xem và quản lý ({len(st.session_state.texts)} tài liệu)"):
            for i, text in enumerate(st.session_state.texts):
                # Lấy tên file từ dòng đầu tiên
                doc_name = text.split('\n')[0].replace("--- Nguồn: ", "").replace(" ---", "")
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.info(f"📄 {doc_name}")
                with col2:
                    if st.button("Xóa", key=f"delete_{i}", use_container_width=True):
                        # Xóa văn bản và cập nhật lại
                        st.session_state.texts.pop(i)
                        st.session_state.index = update_index(st.session_state.texts)
                        save_data(st.session_state.texts, st.session_state.index)
                        st.success("Đã xóa!")
                        st.rerun() # Tải lại trang để cập nhật danh sách