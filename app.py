from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import time
import traceback
from direct_rag import DirectRAG

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Khởi tạo DirectRAG một lần và sử dụng xuyên suốt ứng dụng
rag_system = None


def get_rag_system():
    global rag_system
    if rag_system is None:
        rag_system = DirectRAG()
        # Đảm bảo kết nối được thiết lập
        if hasattr(rag_system, 'vector_store'):
            print("Thiết lập kết nối Vector Store khi khởi tạo...")
            rag_system.vector_store.connect()
            # Tải tất cả collections
            rag_system.collections = rag_system.vector_store.load_collections()
    return rag_system


# Lưu trữ ngữ cảnh hội thoại cho mỗi phiên
conversation_contexts = {}


@app.route('/')
def index():
    """Trả về trang chủ của ứng dụng"""
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def process_query():
    """API endpoint để xử lý câu hỏi từ người dùng"""
    data = request.json
    user_query = data.get('query', '')
    session_id = data.get('session_id', 'default')

    # Kiểm tra nếu không có câu hỏi
    if not user_query:
        return jsonify({
            'status': 'error',
            'message': 'Vui lòng cung cấp câu hỏi'
        }), 400

    try:
        # Lấy hệ thống RAG đã được khởi tạo
        rag = get_rag_system()

        # Kiểm tra kết nối Vector Store và chỉ kết nối lại nếu kết nối đã bị mất
        if hasattr(rag, 'vector_store') and not rag.vector_store.connection_established:
            print("Phát hiện kết nối Vector Store đã bị mất, đang kết nối lại...")
            rag.vector_store.connect()

        # Kiểm tra kết nối SQL và chỉ tạo lại nếu bị mất
        try:
            # Thử thực hiện truy vấn đơn giản để kiểm tra kết nối
            cursor = rag.sql_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        except Exception as e:
            print(f"Phát hiện kết nối SQL đã bị mất, đang kết nối lại... Lỗi: {e}")
            import sqlite3
            from config import DATABASE_PATH
            rag.sql_conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)

        # Đo thời gian xử lý
        start_time = time.time()

        # Ghi nhận mỗi bước xử lý
        processing_steps = {}

        # Bước 1: Xử lý tham chiếu
        resolved_query = rag._resolve_references(user_query)
        processing_steps['resolve_references'] = {
            'time': time.time() - start_time,
            'result': resolved_query
        }

        # Bước 2: Làm rõ câu hỏi
        step_start = time.time()
        clarified_query = rag.clarify_query(resolved_query)
        processing_steps['clarify_query'] = {
            'time': time.time() - step_start,
            'result': clarified_query
        }

        # Bước 3: Phân loại domain
        step_start = time.time()
        query_domain = rag.classify_query_domain(clarified_query)
        processing_steps['classify_domain'] = {
            'time': time.time() - step_start,
            'result': query_domain
        }

        # Xử lý câu hỏi dựa trên domain
        if query_domain == "UNRELATED":
            # Xử lý câu hỏi không liên quan
            step_start = time.time()
            result = rag.handle_unrelated_query(user_query, clarified_query)
            processing_steps['handle_unrelated'] = {
                'time': time.time() - step_start
            }
        else:
            # Bước 4: Phân loại loại câu hỏi
            step_start = time.time()
            query_type = rag.classify_query(clarified_query)
            processing_steps['classify_query'] = {
                'time': time.time() - step_start,
                'result': query_type
            }

            # Bước 5: Xử lý câu hỏi theo loại
            step_start = time.time()
            if query_type == "GENERAL":
                result = rag.handle_general_query(user_query, clarified_query)
                processing_steps['handle_general'] = {
                    'time': time.time() - step_start
                }
            elif query_type == "SPECIFIC-VECTOR":
                result = rag.handle_specific_vector_query(user_query, clarified_query)
                processing_steps['handle_specific_vector'] = {
                    'time': time.time() - step_start
                }
            elif query_type == "SPECIFIC-SQL":
                result = rag.handle_specific_sql_query(user_query, clarified_query)
                processing_steps['handle_specific_sql'] = {
                    'time': time.time() - step_start
                }
            elif query_type == "SPECIFIC-HYBRID":
                result = rag.handle_specific_hybrid_query(user_query, clarified_query)
                processing_steps['handle_specific_hybrid'] = {
                    'time': time.time() - step_start
                }
            else:
                # Loại câu hỏi không xác định, xử lý như câu hỏi chung
                result = rag.handle_general_query(user_query, clarified_query)
                processing_steps['handle_fallback'] = {
                    'time': time.time() - step_start
                }

        # Cập nhật ngữ cảnh hội thoại
        rag._update_conversation_context(result)

        # Tổng thời gian xử lý
        total_time = time.time() - start_time

        # Chuẩn bị phản hồi
        response = {
            'status': 'success',
            'original_query': user_query,
            'clarified_query': clarified_query,
            'query_type': result.get('query_type', 'UNKNOWN'),
            'response': result.get('response', 'Không có phản hồi'),
            'processing_time': f"{total_time:.2f} giây",
            'has_map': result.get('has_map', False)
        }

        # Thêm thông tin chi tiết khác tùy theo loại câu hỏi
        if 'product_name' in result:
            response['product_name'] = result['product_name']

        if 'field' in result:
            response['field'] = result['field']

        if 'fields' in result:
            response['fields'] = result['fields']

        # Debug mode: thêm thông tin về các bước xử lý
        if data.get('debug', False):
            response['debug'] = {
                'processing_steps': processing_steps
            }

            # Thêm thông tin debug khác tùy theo loại câu hỏi
            if 'raw_results' in result:
                response['debug']['raw_results'] = result['raw_results']

            if 'raw_vector_results' in result:
                response['debug']['raw_vector_results'] = result['raw_vector_results']

            if 'raw_sql_results' in result:
                response['debug']['raw_sql_results'] = result['raw_sql_results']

        return jsonify(response)

    except Exception as e:
        # Xử lý lỗi và trả về thông báo lỗi chi tiết
        error_traceback = traceback.format_exc()
        return jsonify({
            'status': 'error',
            'message': f'Lỗi khi xử lý câu hỏi: {str(e)}',
            'traceback': error_traceback if data.get('debug', False) else None
        }), 500


@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """API endpoint để xóa ngữ cảnh hội thoại"""
    data = request.json
    session_id = data.get('session_id', 'default')

    try:
        # Lấy hoặc khởi tạo hệ thống RAG
        rag = get_rag_system()

        # Reset ngữ cảnh hội thoại
        rag.conversation_context = {
            "last_product_name": None,
            "last_query_type": None,
            "last_query": None
        }

        return jsonify({
            'status': 'success',
            'message': 'Đã xóa ngữ cảnh hội thoại'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Lỗi khi xóa ngữ cảnh hội thoại: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint để kiểm tra trạng thái của hệ thống"""
    try:
        # Kiểm tra xem hệ thống RAG có khởi tạo được không
        rag = get_rag_system()

        # Kiểm tra kết nối đến vector store
        vector_store_status = "OK" if rag.vector_store.connect() else "ERROR"

        # Kiểm tra kết nối đến SQL
        sql_status = "OK"
        try:
            cursor = rag.sql_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        except Exception as e:
            sql_status = f"ERROR: {str(e)}"

        return jsonify({
            'status': 'success',
            'message': 'Hệ thống hoạt động bình thường',
            'system_status': {
                'vector_store': vector_store_status,
                'sql': sql_status
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Lỗi khi kiểm tra trạng thái: {str(e)}'
        }), 500


@app.teardown_appcontext
def cleanup(exc):
    """
    Chỉ dọn dẹp tài nguyên khi ứng dụng kết thúc hoàn toàn,
    không đóng kết nối sau mỗi request
    """
    # Không làm gì cả để giữ kết nối mở
    pass


if __name__ == '__main__':
    # Khởi tạo hệ thống RAG trước khi chạy ứng dụng
    rag_system = get_rag_system()
    print("Đã khởi tạo hệ thống RAG thành công")

    try:
        # Chạy ứng dụng Flask
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        # Khi nhận Ctrl+C hoặc tắt server
        print("Nhận tín hiệu tắt server, đang đóng kết nối...")
        if rag_system:
            rag_system.close()
        print("Đã đóng tất cả kết nối. Tạm biệt!")