import json
import os

directories = [
    "C:\\Users\\dangn\\PycharmProjects\\Law\\crawldata\\multi_threads_earphone_product_links.json",
    "C:\\Users\\dangn\\PycharmProjects\\Law\\crawldata\\multi_threads_laptop_product_links.json",
    "C:\\Users\\dangn\\PycharmProjects\\Law\\crawldata\\multi_threads_smartphone_product_links.json",
    "C:\\Users\\dangn\\PycharmProjects\\Law\\crawldata\\multi_threads_speaker_product_links.json",
    "C:\\Users\\dangn\\PycharmProjects\\Law\\crawldata\\multi_threads_tablet_product_links.json",
    "C:\\Users\\dangn\\PycharmProjects\\Law\\crawldata\\multi_threads_watch_product_links.json",
]

for directory in directories:
    try:
        # Kiểm tra nếu file tồn tại và không rỗng
        if os.path.exists(directory) and os.path.getsize(directory) > 0:
            with open(directory, "r", encoding="utf-8") as f:
                data = json.load(f)  # Đọc file JSON
        else:
            print(f"Skipping empty file: {directory}")
            continue

        # Loại bỏ trùng lặp và sắp xếp
        fix_data = sorted(set(data))

        print(f"File: {directory} - Total unique items: {len(fix_data)}")

        # Ghi lại file với dữ liệu đã xử lý
        with open(directory, "w", encoding="utf-8") as f:
            json.dump(fix_data, f, ensure_ascii=False, indent=4)

    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {directory}")
    except Exception as e:
        print(f"Unexpected error processing {directory}: {e}")
