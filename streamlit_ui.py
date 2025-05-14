import streamlit as st
from direct_rag import DirectRAG
import time

# Set page config
st.set_page_config(
    page_title="Tư vấn thiết bị di động",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)


@st.cache_resource
def get_rag_system():
    return DirectRAG()


# Custom CSS for styling with dark theme
st.markdown("""
<style>
    /* Dark background */
    .stApp {
        background-color: #1e1e1e;
        color: white;
    }

    /* Title styling */
    .title-box {
        border: 1px solid #2a2b32;
        border-radius: 8px;
        padding: 15px;
        margin: 10px auto 20px auto;
        text-align: center;
        max-width: 400px;
        background-color: #2a2b32;
    }

    /* Category buttons */
    .category-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        margin: 15px auto;
        max-width: 800px;
        gap: 10px;
    }

    .category-button {
        border: 1px solid #2a2b32;
        border-radius: 20px;
        padding: 8px 15px;
        text-align: center;
        background-color: #2a2b32;
        color: white;
        cursor: pointer;
        min-width: 100px;
    }

    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: #1e1e1e;
        color: white;
        border-radius: 30px;
        padding: 10px 20px;
        border: 1px solid #565869;
        font-size: 16px;
    }

    /* Style for placeholder text */
    .stTextInput > div > div > input::placeholder {
        color: #8e8ea0;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header {
        visibility: hidden;
    }

    /* Chat message styling */
    .stChatMessage {
        background-color: #2a2b32;
        color: white;
        margin-bottom: 10px;
        border-radius: 8px;
    }

    /* Adjust chat container position */
    .stChatMessageContent {
        padding: 8px 12px;
    }

    /* Custom container for chat history */
    .chat-container {
        margin-top: 0;
        padding-top: 0;
    }

    /* Make h2 in title box smaller */
    .title-box h2 {
        margin: 0;
        font-size: 1.5rem;
    }

    /* Style for footer in messages */
    .message-footer {
        font-size: 0.8rem;
        color: #8e8ea0;
        margin-top: 10px;
        border-top: 1px solid #565869;
        padding-top: 5px;
    }

    /* Hide elements when chat is started */
    .hide-on-chat {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.rag = get_rag_system()
if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "current_response" not in st.session_state:
    st.session_state.current_response = ""
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

# Create a container for chat with less vertical space
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Title and category buttons (only shown when chat is not started)
if not st.session_state.chat_started:
    # Title in a box
    st.markdown('<div class="title-box"><h2>Tư vấn thiết bị di động</h2></div>', unsafe_allow_html=True)

    # Category buttons
    st.markdown("""
    <div class="category-container">
        <div class="category-button">Tablet</div>
        <div class="category-button">Laptop</div>
        <div class="category-button">Watch</div>
        <div class="category-button">Phone</div>
        <div class="category-button">Speaker</div>
        <div class="category-button">Earphone</div>
    </div>
    """, unsafe_allow_html=True)

# Display chat history
if st.session_state.chat_started:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])

st.markdown('</div>', unsafe_allow_html=True)

# Chat input
prompt = st.chat_input("Hỏi bất kì điều gì...", key="chat_input")

# Handle input
if prompt:
    if not st.session_state.chat_started:
        st.session_state.chat_started = True

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # # Display chat history
    # for message in st.session_state.messages:
    #     if message["role"] == "user":
    #         with st.chat_message("user"):
    #             st.markdown(message["content"])
    #     else:
    #         with st.chat_message("assistant"):
    #             st.markdown(message["content"])

    # Process response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        st.session_state.streaming = True
        st.session_state.current_response = ""


        # Hàm callback để cập nhật phản hồi theo thời gian thực
        def update_response(new_content):
            st.session_state.current_response += new_content
            message_placeholder.markdown(st.session_state.current_response)


        try:
            # Xử lý câu hỏi người dùng
            start_time = time.time()

            # Lấy đối tượng RAG
            rag = st.session_state.rag

            # Làm rõ câu hỏi
            with st.spinner("Đang làm rõ câu hỏi..."):
                clarified_query = rag.clarify_query(prompt)

            # Phân loại domain
            domain_classification = rag.classify_query_domain(clarified_query)

            if domain_classification == "UNRELATED":
                # Xử lý câu hỏi không liên quan
                result = rag.handle_unrelated_query(prompt, clarified_query)
                # Sử dụng streaming cho phản hồi
                rag.call_openai(
                    f"Trả lời câu hỏi không liên quan: {prompt}\nCâu trả lời gợi ý: {result['response']}",
                    temperature=0.7,
                    stream=True,
                    callback=update_response
                )
            else:
                # Phân loại loại câu hỏi
                query_type = rag.classify_query(clarified_query)

                if query_type == "GENERAL":
                    # Xử lý câu hỏi chung
                    with st.spinner("Đang tìm kiếm thông tin..."):
                        result = rag.handle_general_query(prompt, clarified_query)
                        # Sử dụng streaming cho phản hồi
                        rag.call_openai(
                            f"Trả lời câu hỏi chung: {prompt}\nCâu trả lời gợi ý: {result['response']}",
                            temperature=0.7,
                            stream=True,
                            callback=update_response
                        )

                elif "SPECIFIC" in query_type:
                    # Xử lý câu hỏi cụ thể
                    with st.spinner("Đang tìm kiếm thông tin sản phẩm..."):
                        if query_type == "SPECIFIC-SQL":
                            result = rag.handle_specific_sql_query(prompt, clarified_query)
                        elif query_type == "SPECIFIC-VECTOR":
                            result = rag.handle_specific_vector_query(prompt, clarified_query)
                        elif query_type == "SPECIFIC-HYBRID":
                            result = rag.handle_specific_hybrid_query(prompt, clarified_query)

                        # Sử dụng streaming cho phản hồi
                        rag.call_openai(
                            f"Trả lời câu hỏi cụ thể: {prompt}\nCâu trả lời gợi ý: {result['response']}",
                            temperature=0.7,
                            stream=True,
                            callback=update_response
                        )

            # Tính thời gian xử lý
            response_time = time.time() - start_time

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

            # Thêm phản hồi vào lịch sử
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.current_response})
            st.session_state.streaming = False

        except Exception as e:
            error_message = f"Xảy ra lỗi: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

    # Force rerun to update UI
    st.rerun()

# Add reset button to sidebar
with st.sidebar:
    if st.button("Xóa lịch sử"):
        st.session_state.chat_started = False
        st.session_state.messages = []
        st.rerun()