import json

# formatted_products = []
# seen_product_names = set()  # Để theo dõi tên sản phẩm đã thấy
#
# json_data = [
#     "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\final_data_in_json\\formatted_earphone.json",
#     "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\final_data_in_json\\formatted_laptop.json",
#     "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\final_data_in_json\\formatted_smartphones.json",
#     "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\final_data_in_json\\formatted_speakers.json",
#     "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\final_data_in_json\\formatted_tablets.json",
#     "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\final_data_in_json\\formatted_watches.json"
# ]
#
# for file in json_data:
#     with open(file, "r", encoding="utf-8") as json_file:
#         data = json.load(json_file)
#         for item in data:
#             product_name = item.get("product_name", "")
#             if product_name and product_name not in seen_product_names:
#                 seen_product_names.add(product_name)
#                 formatted_products.append(item)
#
# # Lưu kết quả vào một file mới
# with open("E:\\merged_data.json", "w", encoding="utf-8") as outfile:
#     json.dump(formatted_products, outfile, ensure_ascii=False, indent=4)
#
# print(f"Đã gộp {len(formatted_products)} sản phẩm không trùng lặp vào file merged_data.json")
# 1053 sản phẩm


# def check_data_length(json_file_path):
#     with open("C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\merged_data.json", "r", encoding="utf-8") as f:
#         data = json.load(f)
#
#     # Đếm số lượng item có trường vượt quá 5000 ký tự
#     count_long_items = 0
#     total_items = len(data)
#     long_fields = {}
#
#     for i, item in enumerate(data):
#         has_long_field = False
#         for field in ["product_name", "product_info_summary", "warranty_summary",
#                       "technical_summary", "feature_summary", "content_summary", "full_promotion"]:
#             if field in item and len(item[field]) > 5000:
#                 has_long_field = True
#                 long_fields[field] = long_fields.get(field, 0) + 1
#                 print(f"Item {i}: Field {field} có độ dài {len(item[field])}")
#
#         if has_long_field:
#             count_long_items += 1
#
#     print(f"Tổng số item: {total_items}")
#     print(
#         f"Số item có ít nhất một trường vượt quá 5000 ký tự: {count_long_items} ({count_long_items / total_items * 100:.2f}%)")
#     print("Phân bố các trường vượt giới hạn:")
#     for field, count in long_fields.items():
#         print(f"  - {field}: {count} items ({count / total_items * 100:.2f}%)")
with open("C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\merged_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
count = 0
for idx,item in enumerate(data):
        if(len(item["content_summary"])>5000):
                count +=1
                print(idx)
print(count)