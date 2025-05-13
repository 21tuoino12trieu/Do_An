import argparse
import json
from direct_rag import DirectRAG
import time


def main():
    parser = argparse.ArgumentParser(description="Mobile Device RAG System (Phiên bản cải tiến)")
    parser.add_argument("--query", type=str, help="Câu hỏi người dùng")
    parser.add_argument("--interactive", action="store_true", help="Chạy ở chế độ tương tác")
    parser.add_argument("--debug", action="store_true", help="Hiển thị thông tin gỡ lỗi chi tiết")
    args = parser.parse_args()

    rag_system = DirectRAG()

    try:
        if args.interactive:
            print("=== Hệ thống RAG Thiết bị Di động - Chế độ tương tác ===")
            print("Nhập 'exit' hoặc 'quit' để thoát")
            print("")

            while True:
                user_query = input("\nNhập câu hỏi của bạn: ")
                if user_query.lower() in ['exit', 'quit']:
                    break

                start_time = time.time()
                result = rag_system.process_query(user_query)
                processing_time = time.time() - start_time

                print("\n=== Phản hồi ===")
                print(result['response'])

                if args.debug:
                    print("\n=== Thông tin gỡ lỗi ===")
                    print(f"Câu hỏi gốc: {result['original_query']}")
                    print(f"Câu hỏi đã làm rõ: {result['clarified_query']}")
                    print(f"Loại truy vấn: {result['query_type']}")

                    if 'product_name' in result:
                        print(f"Sản phẩm: {result['product_name']}")

                    if 'field' in result:
                        print(f"Trường dữ liệu: {result['field']}")

                    if 'fields' in result:
                        print(f"Các trường dữ liệu: {result['fields']}")

                print(f"\nThời gian xử lý: {processing_time:.2f} giây")

        elif args.query:
            start_time = time.time()
            result = rag_system.process_query(args.query)
            processing_time = time.time() - start_time

            output = {
                "response": result['response'],
                "query_type": result['query_type'],
                "clarified_query": result['clarified_query'],
                "processing_time": f"{processing_time:.2f} giây"
            }

            if 'product_name' in result:
                output["product_name"] = result['product_name']

            if 'field' in result:
                output["field"] = result['field']

            if 'fields' in result:
                output["fields"] = result['fields']

            if args.debug:
                if 'raw_results' in result:
                    output["raw_results"] = result['raw_results']
                if 'raw_vector_results' in result:
                    output["raw_vector_results"] = result['raw_vector_results']
                if 'raw_sql_results' in result:
                    output["raw_sql_results"] = result['raw_sql_results']

            print(json.dumps(output, indent=2, ensure_ascii=False))

        else:
            print("Vui lòng cung cấp câu hỏi với --query hoặc sử dụng chế độ tương tác với --interactive")

    finally:
        rag_system.close()


if __name__ == "__main__":
    main()