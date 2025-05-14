from vector_store import VectorStore
import sqlite3
from openai import OpenAI
import google.generativeai as genai
from config import DATABASE_PATH
from prompts import (
    QUERY_CLARIFICATION_PROMPT,
    QUERY_CLASSIFICATION_PROMPT,
    QUERY_DOMAIN_CLASSIFICATION_PROMPT,
    QUERY_UNRELATED_HANDLE_PROMPT,
    GENERAL_FIELD_IDENTIFICATION_PROMPT,
    EXTRACT_PRODUCT_NAME_PROMPT,
    SPECIFIC_FIELDS_IDENTIFICATION_PROMPT,
    SQL_GENERATION_PROMPT,
    GENERAL_RESPONSE_GENERATION_PROMPT,
    SPECIFIC_VECTOR_RESPONSE_PROMPT,
    SPECIFIC_SQL_RESPONSE_PROMPT,
)


class DirectRAG:
    """
    Hệ thống RAG gọi trực tiếp qua OpenAI Client - phiên bản cải tiến
    """

    def __init__(self):
        # Khởi tạo kết nối với SQLite
        self.sql_conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)

        # Khởi tạo Vector Store
        self.vector_store = VectorStore()
        self.collections = self.vector_store.load_collections()
        self.embedding_model = self.vector_store._load_embedding_model()
        self.reranker = self.vector_store.reranker

        genai.configure(api_key="AIzaSyBLEHXcGtxM5EKA2vE53ooMddz2ELYVatM")

        # Khởi tạo OpenAI client
        self.client = OpenAI(
            api_key="sk-DA3F9KX5B3sWaspdA1B9Cc9069Ab416fAa01C50316Ac17Ff",
            base_url="https://api.sv2.llm.ai.vn/v1"
        )

        print("DirectRAG đã khởi tạo thành công")

    def call_openai(self, prompt, temperature=0.1, model="gpt-4o", stream=False, callback=None):
        """Gọi OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=stream,
            )

            if stream and callback:
                # Xử lý streaming nếu được yêu cầu
                full_response = ""
                for chunk in response:
                    if chunk.choices and len(chunk.choices) > 0:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_response += content
                            callback(content)
                return full_response
            else:
                # Xử lý không streaming như trước
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Lỗi khi gọi OpenAI API: {e}")
            return f"Không thể nhận phản hồi từ AI: {str(e)}"

        # try:
        #     # Cấu hình model
        #     generation_config = {
        #         "temperature": temperature,
        #         "top_p": 0.95,
        #         "top_k": 0,
        #         "max_output_tokens": 2048,
        #     }
        #
        #     # Khởi tạo model
        #     model = genai.GenerativeModel(model_name=model,
        #                                   generation_config=generation_config)
        #
        #     # Gọi API
        #     response = model.generate_content(prompt)
        #
        #     # Trả về kết quả
        #     return response.text
        # except Exception as e:
        #     print(f"Lỗi khi gọi Gemini API: {e}")
        #     return f"Không thể nhận phản hồi từ AI: {str(e)}"

    def clarify_query(self, user_query):
        """Làm rõ câu hỏi người dùng"""
        prompt = QUERY_CLARIFICATION_PROMPT.format(user_query=user_query)
        return self.call_openai(prompt, temperature=0.1)

    def classify_query_domain(self, clarified_query):
        """Phân loại câu hỏi thuộc domain thiết bị di động hay không"""
        formatted_prompt = QUERY_DOMAIN_CLASSIFICATION_PROMPT.format(clarified_query=clarified_query)
        response = self.call_openai(formatted_prompt, temperature=0.1)

        # Đảm bảo kết quả trả về là một trong hai giá trị hợp lệ
        response = response.strip().upper()
        if response not in ["RELATED", "UNRELATED"]:
            # Mặc định là RELATED nếu không xác định được rõ ràng
            return "RELATED"

        return response

    def handle_unrelated_query(self, original_query, clarified_query):
        """Xử lý câu hỏi không liên quan đến thiết bị di động"""
        prompt = QUERY_UNRELATED_HANDLE_PROMPT.format(clarified_query=clarified_query)
        response = self.call_openai(prompt, temperature=1)

        return {
            "original_query": original_query,
            "clarified_query": clarified_query,
            "query_type": "UNRELATED",
            "response": response
        }

    def classify_query(self, clarified_query):
        """Phân loại câu hỏi theo cấu trúc mới: GENERAL hoặc SPECIFIC-XXX"""
        prompt = QUERY_CLASSIFICATION_PROMPT.format(clarified_query=clarified_query)
        initial_classification = self.call_openai(prompt, temperature=0.1)
        return initial_classification

    def identify_general_field(self, clarified_query):
        """Xác định trường tối ưu nhất cho câu hỏi chung chung"""
        prompt = GENERAL_FIELD_IDENTIFICATION_PROMPT.format(clarified_query=clarified_query)
        return self.call_openai(prompt, temperature=0.1).strip()

    def extract_product_name(self, clarified_query):
        """Trích xuất tên sản phẩm từ câu hỏi cụ thể"""
        prompt = EXTRACT_PRODUCT_NAME_PROMPT.format(clarified_query=clarified_query)
        return self.call_openai(prompt, temperature=0.1).strip()

    def identify_specific_fields(self, clarified_query):
        """Xác định các trường cần lấy cho câu hỏi về sản phẩm cụ thể"""
        prompt = SPECIFIC_FIELDS_IDENTIFICATION_PROMPT.format(clarified_query=clarified_query)
        response = self.call_openai(prompt, temperature=0.1).strip()

        if response.lower() == "all":
            return ["product_info", "warranty", "technical", "feature", "content", "full_promotion"]
        else:
            return [field.strip() for field in response.split(",")]

    def execute_sql_query(self, clarified_query, product_name):
        """Thực hiện truy vấn SQL"""
        # Tạo câu lệnh SQL từ câu hỏi và tên sản phẩm
        prompt = SQL_GENERATION_PROMPT.format(
            clarified_query=clarified_query,
            product_name=product_name
        )
        sql_query = self.call_openai(prompt, temperature=0.1)

        # Loại bỏ các định dạng không cần thiết nếu có
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].strip()

        print(f"Generated SQL query: {sql_query}")

        # Thực thi câu lệnh SQL
        cursor = self.sql_conn.cursor()
        try:
            cursor.execute(sql_query)
            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return results
        except Exception as e:
            error_message = f"SQL Error: {str(e)}"
            print(error_message)
            return []
        finally:
            cursor.close()

    def perform_semantic_search(self, query, field_name, top_k=5):
        """Thực hiện tìm kiếm ngữ nghĩa trên một trường cụ thể"""
        try:
            print(f"\n🔍 Tìm kiếm ngữ nghĩa trong {field_name} với query: {query}")

            # Tạo embedding cho query
            embedding_result = self.embedding_model.encode(query, max_length=2048)
            query_embedding = embedding_result["dense_vecs"]

            # Đảm bảo collection được tải
            if field_name not in self.collections:
                print(f"Field {field_name} không tồn tại trong collections")
                return []

            collection = self.collections[field_name]
            search_params = {"metric_type": "IP", "params": {"ef": 1024}}

            # Thực hiện tìm kiếm với embedding gốc
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["product_name", "chunk_id", "text_data"]
            )

            formatted_results = []
            for hit in results[0]:
                formatted_results.append({
                    "product_name": hit.entity.get("product_name"),
                    "chunk_id": hit.entity.get("chunk_id"),
                    "text_data": hit.entity.get("text_data"),
                    "score": hit.score
                })

            print(f"✅ Tìm thấy {len(formatted_results)} kết quả từ {field_name}")
            return formatted_results

        except Exception as e:
            print(f"❌ Lỗi trong perform_semantic_search: {e}")
            return []

    def handle_general_query(self, original_query, clarified_query):
        """Xử lý câu hỏi chung chung về nhiều sản phẩm"""
        # 1. Xác định trường tối ưu để tìm kiếm
        optimal_field = self.identify_general_field(clarified_query)
        print(f"Trường tối ưu cho câu hỏi chung: {optimal_field}")

        # 2. Thực hiện tìm kiếm ngữ nghĩa
        search_results = self.perform_semantic_search(clarified_query, optimal_field, top_k=5)

        if not search_results:
            return {
                "original_query": original_query,
                "clarified_query": clarified_query,
                "query_type": "GENERAL",
                "response": f"Xin lỗi, tôi không tìm thấy sản phẩm nào phù hợp với yêu cầu của bạn.",
                "field": optimal_field,
                "raw_results": []
            }

        # 3. Tổ chức kết quả để hiển thị
        formatted_results = ""
        for i, result in enumerate(search_results):
            formatted_results += f"--- Sản phẩm {i + 1}: {result['product_name']} ---\n"
            formatted_results += f"Thông tin: {result['text_data']}\n\n"

        # 4. Tạo câu trả lời từ kết quả tìm kiếm
        response = self.call_openai(
            GENERAL_RESPONSE_GENERATION_PROMPT.format(
                original_query=original_query,
                clarified_query=clarified_query,
                field_name=optimal_field,
                results=formatted_results
            ),
            temperature=1
        )

        return {
            "original_query": original_query,
            "clarified_query": clarified_query,
            "query_type": "GENERAL",
            "response": response,
            "field": optimal_field,
            "raw_results": search_results
        }

    def handle_specific_vector_query(self, original_query, clarified_query):
        """Xử lý câu hỏi về một sản phẩm cụ thể (dùng vector search)"""
        # 1. Trích xuất tên sản phẩm từ câu hỏi
        product_name = self.extract_product_name(clarified_query)
        print(f"Tên sản phẩm được trích xuất: {product_name}")

        # 2. Xác định các trường thông tin cần lấy
        fields = self.identify_specific_fields(clarified_query)
        print(f"Các trường thông tin cần lấy: {fields}")

        # 3. Tìm kiếm sản phẩm theo tên để xác nhận
        product_search_results = self.perform_semantic_search(product_name, "product_name", top_k=3)

        if not product_search_results:
            return {
                "original_query": original_query,
                "clarified_query": clarified_query,
                "query_type": "SPECIFIC-VECTOR",
                "response": f"Xin lỗi, tôi không tìm thấy thông tin về sản phẩm {product_name}.",
                "product_name": product_name,
                "fields": fields,
                "raw_results": []
            }

        # 4. Chọn sản phẩm phù hợp nhất (điểm cao nhất)
        best_match = product_search_results[0]
        actual_product_name = best_match["product_name"]
        print(f"Sản phẩm phù hợp nhất: {actual_product_name}")

        # 5. Thu thập thông tin từ các trường dữ liệu cần thiết
        field_results = {}
        for field in fields:
            try:
                collection = self.collections[field]

                # Truy vấn trực tiếp sử dụng tên sản phẩm chính xác
                query_expr = f'product_name == "{actual_product_name}"'  # Điều kiện chính xác
                print(f"Truy vấn trực tiếp collection {field} với điều kiện: {query_expr}")

                results = collection.query(
                    expr=query_expr,
                    output_fields=["product_name", "chunk_id", "text_data"]
                )

                print(f"Tìm thấy {len(results)} kết quả từ collection {field}")

                if results:
                    # Lấy dữ liệu từ kết quả đầu tiên
                    field_results[field] = results[0]["text_data"]
                else:
                    print(f"Không tìm thấy kết quả nào cho {field} với product_name={actual_product_name}")

                    # Thử phương pháp thay thế: tìm gần đúng (nếu cần)
                    fallback_expr = f'product_name like "%{actual_product_name}%"'
                    fallback_results = collection.query(
                        expr=fallback_expr,
                        output_fields=["product_name", "chunk_id", "text_data"],
                        limit=1
                    )

                    if fallback_results:
                        field_results[field] = fallback_results[0]["text_data"]
                        print(f"Tìm thấy kết quả gần đúng: {fallback_results[0]['product_name']}")
                    else:
                        field_results[field] = f"Không tìm thấy thông tin {field} cho sản phẩm {actual_product_name}"

            except Exception as e:
                print(f"Lỗi khi truy vấn collection {field}: {e}")
                field_results[field] = f"Không thể lấy thông tin {field} (lỗi: {str(e)})"

        # 6. Tổ chức kết quả để hiển thị
        formatted_results = f"--- Thông tin sản phẩm: {actual_product_name} ---\n\n"
        for field, text in field_results.items():
            field_display = {
                "product_info": "Thông tin sản phẩm",
                "warranty": "Bảo hành",
                "technical": "Thông số kỹ thuật",
                "feature": "Tính năng nổi bật",
                "content": "Mô tả chi tiết",
                "full_promotion": "Khuyến mãi"
            }.get(field, field.replace("_", " ").title())

            formatted_results += f"### {field_display}:\n{text}\n\n"

        # 7. Tạo câu trả lời từ kết quả tìm kiếm
        response = self.call_openai(
            SPECIFIC_VECTOR_RESPONSE_PROMPT.format(
                original_query=original_query,
                clarified_query=clarified_query,
                product_name=actual_product_name,
                fields=", ".join(fields),
                results=formatted_results
            ),
            temperature=0.1
        )

        return {
            "original_query": original_query,
            "clarified_query": clarified_query,
            "query_type": "SPECIFIC-VECTOR",
            "response": response,
            "product_name": actual_product_name,
            "fields": fields,
            "raw_results": field_results
        }

    def handle_specific_sql_query(self, original_query, clarified_query):
        """Xử lý câu hỏi về một sản phẩm cụ thể (dùng SQL query)"""
        # 1. Trích xuất tên sản phẩm từ câu hỏi
        product_name = self.extract_product_name(clarified_query)
        print(f"Tên sản phẩm được trích xuất (SQL): {product_name}")

        # 2. Xác định loại câu hỏi SQL (về địa chỉ, giá, hoặc số lượng sản phẩm)
        sql_query_type = self._identify_sql_query_type(clarified_query)
        print(f"Loại câu hỏi SQL: {sql_query_type}")

        # 3. Thực hiện truy vấn SQL
        sql_results = self.execute_sql_query(clarified_query, product_name)

        if not sql_results:
            return {
                "original_query": original_query,
                "clarified_query": clarified_query,
                "query_type": "SPECIFIC-SQL",
                "sql_query_type": sql_query_type,  # Thêm loại câu hỏi SQL cụ thể
                "response": f"Xin lỗi, tôi không tìm thấy thông tin {self._get_sql_type_description(sql_query_type)} sản phẩm {product_name}.",
                "product_name": product_name,
                "raw_results": []
            }

        # 4. Tổ chức kết quả SQL để hiển thị
        formatted_results = self._format_sql_results(sql_query_type, product_name, sql_results)

        # 5. Tạo câu trả lời từ kết quả SQL
        response = self.call_openai(
            SPECIFIC_SQL_RESPONSE_PROMPT.format(
                original_query=original_query,
                clarified_query=clarified_query,
                product_name=product_name,
                results=formatted_results
            ),
            temperature=0.7
        )

        # 6. Kiểm tra xem có bản đồ không để báo cho giao diện người dùng
        has_map = any("map" in result and result["map"] and result["map"] != "None" for result in sql_results)

        return {
            "original_query": original_query,
            "clarified_query": clarified_query,
            "query_type": "SPECIFIC-SQL",
            "sql_query_type": sql_query_type,  # Thêm loại câu hỏi SQL cụ thể
            "response": response,
            "product_name": product_name,
            "has_map": has_map,  # Thêm flag cho biết có bản đồ hay không
            "raw_results": sql_results
        }

    def _identify_sql_query_type(self, query):
        """Xác định loại câu hỏi SQL: địa chỉ, giá hoặc số lượng sản phẩm"""
        prompt = """
        Phân loại câu hỏi sau vào một trong các loại:

        "{query}"
        Các loại:
        1. ADDRESS: Câu hỏi về địa chỉ, cửa hàng, nơi bán
        2. PRICE: Câu hỏi về giá cả của một sản phẩm cụ thể

        Trả về chỉ một trong ba giá trị: ADDRESS, PRICE 
        """
        result = self.call_openai(prompt.format(query=query), temperature=0.1)
        result = result.strip().upper()
        if result not in ["ADDRESS", "PRICE"]:
            # Mặc định là ADDRESS nếu không xác định được
            return "ADDRESS"
        return result

    def _get_sql_type_description(self, sql_query_type):
        """Trả về mô tả cho loại câu hỏi SQL"""
        descriptions = {
            "ADDRESS": "về địa điểm bán",
            "PRICE": "về giá",
            "COUNT": "về số lượng"
        }
        return descriptions.get(sql_query_type, "")

    def _format_sql_results(self, sql_query_type, product_name, sql_results):
        """Format kết quả SQL dựa trên loại câu hỏi"""
        if sql_query_type == "ADDRESS":
            # Format kết quả cho câu hỏi về địa chỉ
            formatted_results = f"--- Thông tin địa điểm bán sản phẩm: {product_name} ---\n\n"

            # Gom nhóm theo sản phẩm
            products_info = {}
            for result in sql_results:
                product = result.get("product_name", "")
                if product not in products_info:
                    products_info[product] = {
                        "price": result.get("price", "N/A"),
                        "locations": []
                    }

                if "address" in result and result["address"]:
                    location = {
                        "address": result["address"],
                        "has_map": result.get("map") and result.get("map") != "None"
                    }
                    products_info[product]["locations"].append(location)

            # Format kết quả
            for product, info in products_info.items():
                formatted_results += f"Sản phẩm: {product}\n"
                formatted_results += f"Giá: {info['price']}\n\n"

                if info["locations"]:
                    formatted_results += "Địa điểm bán:\n"
                    for i, location in enumerate(info["locations"]):
                        formatted_results += f"{i + 1}. {location['address']}"
                        if location.get("has_map"):
                            formatted_results += " (Có bản đồ)"
                        formatted_results += "\n"
                else:
                    formatted_results += "Không có thông tin về địa điểm bán.\n"

        elif sql_query_type == "PRICE":
            # Format kết quả cho câu hỏi về giá
            formatted_results = f"--- Thông tin giá sản phẩm: {product_name} ---\n\n"

            # Lấy giá duy nhất cho sản phẩm
            unique_products = {}
            for result in sql_results:
                product = result.get("product_name", "")
                if product not in unique_products:
                    unique_products[product] = result.get("price", "N/A")

            for product, price in unique_products.items():
                formatted_results += f"Sản phẩm: {product}\n"
                formatted_results += f"Giá: {price}\n\n"

        elif sql_query_type == "COUNT":
            # Format kết quả cho câu hỏi về số lượng sản phẩm theo khoảng giá
            formatted_results = f"--- Danh sách sản phẩm theo yêu cầu ---\n\n"
            formatted_results += f"Số lượng sản phẩm tìm thấy: {len(sql_results)}\n\n"

            # Gom nhóm theo category nếu cần
            categories = {}
            for result in sql_results:
                category = result.get("category", "Không xác định")
                if category not in categories:
                    categories[category] = []

                categories[category].append({
                    "product_name": result.get("product_name", ""),
                    "price": result.get("price", "N/A")
                })

            # Format kết quả theo category
            for category, products in categories.items():
                category_display = {
                    "smartphone": "Điện thoại",
                    "tablet": "Máy tính bảng",
                    "laptop": "Laptop",
                    "earphone": "Tai nghe",
                    "speaker": "Loa",
                    "watch": "Đồng hồ"
                }.get(category, category)

                formatted_results += f"--- {category_display} ---\n"
                for i, product in enumerate(products):
                    formatted_results += f"{i + 1}. {product['product_name']} - {product['price']}\n"
                formatted_results += "\n"

        else:
            # Format mặc định
            formatted_results = f"--- Kết quả tìm kiếm cho: {product_name} ---\n\n"
            for i, result in enumerate(sql_results):
                formatted_results += f"Kết quả {i + 1}:\n"
                for key, value in result.items():
                    if key != "map":  # Không hiển thị mã HTML của bản đồ
                        formatted_results += f"{key}: {value}\n"
                formatted_results += "\n"

        return formatted_results

    def handle_specific_hybrid_query(self, original_query, clarified_query):
        """Xử lý câu hỏi kết hợp cả vector search và SQL query"""
        # 1. Trích xuất tên sản phẩm từ câu hỏi
        product_name = self.extract_product_name(clarified_query)
        print(f"Tên sản phẩm được trích xuất (HYBRID): {product_name}")

        # 2. Xác định các trường thông tin cần lấy
        fields = self.identify_specific_fields(clarified_query)
        print(f"Các trường thông tin cần lấy (HYBRID): {fields}")

        # 3. Thực hiện song song cả 2 loại truy vấn

        # 3.1. Tìm kiếm vector
        product_search_results = self.perform_semantic_search(product_name, "product_name", top_k=3)

        if not product_search_results:
            return {
                "original_query": original_query,
                "clarified_query": clarified_query,
                "query_type": "SPECIFIC-HYBRID",
                "response": f"Xin lỗi, tôi không tìm thấy thông tin về sản phẩm {product_name}.",
                "product_name": product_name,
                "fields": fields,
                "raw_results": {}
            }

        best_match = product_search_results[0]
        actual_product_name = best_match["product_name"]
        print(f"Sản phẩm phù hợp nhất (HYBRID): {actual_product_name}")

        # Thu thập thông tin từ các trường vector
        # Thu thập thông tin từ các trường vector bằng query trực tiếp
        field_results = {}
        for field in fields:
            try:
                collection = self.collections[field]

                query_expr = f'product_name == "{actual_product_name}"'
                print(f"Truy vấn hybrid trực tiếp collection {field}: {query_expr}")

                results = collection.query(
                    expr=query_expr,
                    output_fields=["product_name", "chunk_id", "text_data"]
                )

                if results:
                    field_results[field] = results[0]["text_data"]
                    print(f"Tìm thấy thông tin {field} cho {actual_product_name}")
                else:
                    print(f"Không tìm thấy thông tin {field} cho {actual_product_name}")

                    # Thử fallback dùng semantic search nếu cần
                    field_search = self.perform_semantic_search(actual_product_name, field, top_k=3)
                    for result in field_search:
                        if result["product_name"] == actual_product_name:
                            field_results[field] = result["text_data"]
                            print(f"Tìm thấy thông tin {field} bằng semantic search")
                            break

                    if field not in field_results:
                        field_results[field] = f"Không tìm thấy thông tin {field} cho sản phẩm {actual_product_name}"
            except Exception as e:
                print(f"Lỗi truy vấn hybrid collection {field}: {e}")
                field_results[field] = f"Không thể lấy thông tin {field}"

        # 3.2. Thực hiện truy vấn SQL
        sql_results = self.execute_sql_query(clarified_query, actual_product_name)

        # 4. Tổ chức kết quả vector để hiển thị
        vector_formatted_results = f"--- Thông tin sản phẩm: {actual_product_name} ---\n\n"
        for field, text in field_results.items():
            field_display = {
                "product_info": "Thông tin sản phẩm",
                "warranty": "Bảo hành",
                "technical": "Thông số kỹ thuật",
                "feature": "Tính năng nổi bật",
                "content": "Mô tả chi tiết",
                "full_promotion": "Khuyến mãi"
            }.get(field, field.replace("_", " ").title())

            vector_formatted_results += f"### {field_display}:\n{text}\n\n"

        # 5. Tổ chức kết quả SQL để hiển thị
        sql_formatted_results = ""
        if sql_results:
            sql_formatted_results = f"--- Thông tin giá/địa điểm của sản phẩm: {actual_product_name} ---\n\n"

            # Gom nhóm theo sản phẩm
            products_info = {}
            for result in sql_results:
                product = result.get("product_name", "")
                if product not in products_info:
                    products_info[product] = {
                        "price": result.get("price", "N/A"),
                        "locations": []
                    }

                if "address" in result and result["address"]:
                    location = {
                        "address": result["address"],
                        "map": result.get("map", "")
                    }
                    products_info[product]["locations"].append(location)

            # Format kết quả cho từng sản phẩm
            for product, info in products_info.items():
                sql_formatted_results += f"Giá: {info['price']}\n\n"

                if info["locations"]:
                    sql_formatted_results += "Địa điểm bán:\n"
                    for i, location in enumerate(info["locations"]):
                        sql_formatted_results += f"{i + 1}. {location['address']}\n"
                else:
                    sql_formatted_results += "Không có thông tin về địa điểm bán.\n"
        else:
            sql_formatted_results = "Không tìm thấy thông tin về giá hoặc địa điểm bán sản phẩm này."

        # PHẦN CẦN CẢI THIỆN - Phân tích câu hỏi để xác định trọng số ưu tiên
        vector_keywords = ["thông số", "tính năng", "đánh giá", "bảo hành", "khuyến mãi", "review", "có gì hay"]
        sql_keywords = ["giá", "cửa hàng", "địa điểm", "bán ở đâu", "mua ở đâu", "bao nhiêu tiền"]

        # Xác định độ ưu tiên dựa trên vị trí xuất hiện trong câu hỏi
        vector_priority = False
        sql_priority = False

        # Xác định thứ tự ưu tiên dựa trên vị trí xuất hiện từ khóa đầu tiên
        vector_position = float('inf')
        sql_position = float('inf')

        for keyword in vector_keywords:
            pos = clarified_query.lower().find(keyword)
            if pos != -1 and pos < vector_position:
                vector_position = pos
                vector_priority = True

        for keyword in sql_keywords:
            pos = clarified_query.lower().find(keyword)
            if pos != -1 and pos < sql_position:
                sql_position = pos
                sql_priority = True

        # Xác định thứ tự ưu tiên (thứ tự xuất hiện trước được ưu tiên)
        priority_instruction = ""
        if vector_priority and sql_priority:
            if vector_position < sql_position:
                priority_instruction = "Trong câu trả lời, hãy ưu tiên thông tin về đặc điểm, tính năng sản phẩm trước, sau đó mới đến thông tin về giá và địa điểm bán."
            else:
                priority_instruction = "Trong câu trả lời, hãy ưu tiên thông tin về giá và địa điểm bán trước, sau đó mới đến thông tin về đặc điểm, tính năng sản phẩm."

        # Xác định xem có bản đồ không
        has_map = any("map" in result and result["map"] and result["map"] != "None" for result in sql_results)
        map_instruction = "Cuối cùng, hãy nhắc người dùng rằng họ có thể xem vị trí cửa hàng trên bản đồ bên dưới." if has_map else ""

        # 6. Tạo câu trả lời kết hợp với hướng dẫn ưu tiên
        prompt = f"""
        Tạo câu trả lời kết hợp thông tin sản phẩm và giá/địa điểm bán:

        Câu hỏi gốc: "{original_query}"
        Câu hỏi đã làm rõ: "{clarified_query}"
        Sản phẩm: {actual_product_name}
        Các trường thông tin: {", ".join(fields)}

        Kết quả Vector:
        {vector_formatted_results}

        Kết quả SQL:
        {sql_formatted_results}

        {priority_instruction}

        Hãy tạo một câu trả lời toàn diện kết hợp cả thông tin về sản phẩm và giá/địa điểm bán. Câu trả lời cần:
        1. Mở đầu bằng việc xác nhận sản phẩm người dùng đang hỏi
        2. Cung cấp thông tin chi tiết về sản phẩm theo yêu cầu trong câu hỏi
        3. Cung cấp thông tin về giá (nếu có trong kết quả SQL)
        4. Liệt kê các địa điểm bán (nếu có trong kết quả SQL)
        5. Kết thúc bằng một tóm tắt ngắn gọn

        {map_instruction}

        Câu trả lời phải được tổ chức tốt, hữu ích và đáp ứng đầy đủ yêu cầu của người dùng. Tối đa 800 từ.
        """

        response = self.call_openai(prompt, temperature=0.7)

        return {
            "original_query": original_query,
            "clarified_query": clarified_query,
            "query_type": "SPECIFIC-HYBRID",
            "response": response,
            "product_name": actual_product_name,
            "fields": fields,
            "has_map": has_map,  # Thêm flag cho biết có bản đồ hay không
            "raw_vector_results": field_results,
            "raw_sql_results": sql_results
        }

    def process_query(self, user_query):
        """Xử lý câu hỏi người dùng - điểm vào chính của hệ thống"""
        print(f"Câu hỏi đầu vào người dùng: {user_query}")

        try:
            # 1. Làm rõ câu hỏi
            clarified_query = self.clarify_query(user_query)
            print(f"Câu hỏi đã làm rõ: {clarified_query}")

            # 2. Phân loại domain (RELATED/UNRELATED)
            query_domain = self.classify_query_domain(clarified_query)
            print(f"Domain truy vấn: {query_domain}")

            if query_domain == "UNRELATED":
                return self.handle_unrelated_query(user_query, clarified_query)

            # 3. Phân loại câu hỏi theo cấu trúc mới
            query_type = self.classify_query(clarified_query)
            print(f"Loại truy vấn: {query_type}")

            # 4. Xử lý theo loại câu hỏi
            if query_type == "GENERAL":
                # Câu hỏi chung chung về nhiều sản phẩm
                return self.handle_general_query(user_query, clarified_query)

            elif query_type == "SPECIFIC-VECTOR":
                # Câu hỏi về một sản phẩm cụ thể, cần thông tin từ vector store
                return self.handle_specific_vector_query(user_query, clarified_query)

            elif query_type == "SPECIFIC-SQL":
                # Câu hỏi về một sản phẩm cụ thể, cần thông tin từ SQL (giá, địa điểm)
                return self.handle_specific_sql_query(user_query, clarified_query)

            elif query_type == "SPECIFIC-HYBRID":
                # Câu hỏi về một sản phẩm cụ thể, cần kết hợp cả hai loại thông tin
                return self.handle_specific_hybrid_query(user_query, clarified_query)

            else:
                # Loại truy vấn không xác định, mặc định xử lý như câu hỏi chung
                print(f"Loại truy vấn không xác định: {query_type}, xử lý mặc định")
                return self.handle_general_query(user_query, clarified_query)

        except Exception as e:
            error_message = f"Đã xảy ra lỗi khi xử lý truy vấn: {str(e)}"
            print(error_message)
            return {
                "original_query": user_query,
                "clarified_query": user_query,
                "query_type": "ERROR",
                "response": f"Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi của bạn: {str(e)}. Vui lòng thử lại hoặc đặt câu hỏi khác.",
                "raw_results": error_message
            }

    def close(self):
        """Đóng tất cả các kết nối"""
        if hasattr(self, "sql_conn") and self.sql_conn:
            self.sql_conn.close()
            print("Đã đóng kết nối SQL")

        try:
            # Đóng kết nối Milvus nếu có
            from pymilvus import connections
            connections.disconnect("default")
            print("Đã đóng kết nối Milvus")
        except Exception as e:
            print(f"Lỗi khi đóng kết nối Milvus: {e}")
