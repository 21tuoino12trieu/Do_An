import streamlit as st
from direct_rag import DirectRAG
import re
import time

st.set_page_config(
    page_title="Assistant Tư Vấn Thiết Bị Di Động",
    page_icon="📱",
    layout="wide"
)


@st.cache_resource
def get_rag_system():
    return DirectRAG()


# Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.rag = get_rag_system()

st.title("🤖 Assistant Tư Vấn Thiết Bị Di Động")
st.subheader("Hãy hỏi tôi về các sản phẩm điện tử!")

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Kiểm tra xem có mã nhúng bản đồ không và hiển thị nó
        if message["role"] == "assistant" and "<iframe" in message["content"]:
            map_html = re.search(r'(<iframe.*?</iframe>)', message["content"], re.DOTALL)
            if map_html:
                st.components.v1.html(map_html.group(1), height=400)

# Nhận input từ người dùng
prompt = st.chat_input("Hỏi tôi bất cứ điều gì về thiết bị di động...")

if prompt:
    # Thêm câu hỏi người dùng vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Hiển thị câu hỏi người dùng
    with st.chat_message("user"):
        st.markdown(prompt)

    # Hiển thị trạng thái đang xử lý
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Đang suy nghĩ...")

        try:
            # Xử lý câu hỏi người dùng
            start_time = time.time()
            result = st.session_state.rag.process_query(prompt)
            response_time = time.time() - start_time

            # Lấy phản hồi từ kết quả
            response = result["response"]

            # Tạo thông tin hiển thị ở footer
            query_type_display = {
                "GENERAL": "Câu hỏi chung về nhiều sản phẩm",
                "SPECIFIC-VECTOR": "Câu hỏi về thông tin sản phẩm cụ thể",
                "SPECIFIC-SQL": "Câu hỏi về giá/địa điểm bán",
                "SPECIFIC-HYBRID": "Câu hỏi kết hợp thông tin và giá/địa điểm",
                "ERROR": "Lỗi xử lý"
            }

            query_type = result.get("query_type", "ERROR")
            query_type_text = query_type_display.get(query_type, query_type)

            # Tạo thông tin bổ sung dựa trên loại truy vấn
            additional_info = ""
            if query_type == "GENERAL":
                if "field" in result:
                    field_display = {
                        "product_name": "Tên sản phẩm",
                        "product_info": "Thông tin sản phẩm",
                        "warranty": "Bảo hành",
                        "technical": "Thông số kỹ thuật",
                        "feature": "Tính năng nổi bật",
                        "content": "Mô tả chi tiết",
                        "full_promotion": "Khuyến mãi"
                    }.get(result["field"], result["field"])
                    additional_info = f" • Trường tìm kiếm: {field_display}"
            elif "SPECIFIC" in query_type:
                if "product_name" in result:
                    additional_info = f" • Sản phẩm: {result['product_name']}"

            footer = f"\n\n---\n*Thời gian xử lý: {response_time:.2f} giây • Loại truy vấn: {query_type_text}{additional_info}*"

            # Thêm footer vào phản hồi
            full_response = response + footer

            # Cập nhật placeholder với câu trả lời đầy đủ
            message_placeholder.markdown(full_response)

            # Thêm câu trả lời vào lịch sử
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_message = f"Xảy ra lỗi: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# Thanh bên để hiển thị thông tin bổ sung
with st.sidebar:
    st.header("Thông tin")
    st.write("Hệ thống RAG tư vấn thiết bị di động")
    st.write("Bạn có thể hỏi về:")

    st.subheader("Câu hỏi chung chung")
    st.write("- Tư vấn tai nghe chống ồn tốt")
    st.write("- Tư vấn điện thoại chụp ảnh đẹp")
    st.write("- Đề xuất laptop cho sinh viên IT")

    st.subheader("Câu hỏi cụ thể")
    st.write("- Thông số kỹ thuật của iPhone 13 Pro Max")
    st.write("- Giá của Samsung Galaxy S22 Ultra")
    st.write("- Địa điểm bán tai nghe SoundPeats T3 Pro")
    st.write("- Apple Watch Series 7 có tính năng gì nổi bật?")

    st.divider()

    if st.button("Xóa lịch sử chat", type="primary"):
        st.session_state.messages = []
        st.rerun()