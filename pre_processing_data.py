import json
import re


# --- Đoạn code này dùng để tiền xử lí dữ liệu cho data, bao gồm việc như sau:
# Nối các data rời với nhau, loại bỏ các data nào mà địa chỉ không có vì nếu không tồn tại địa chỉ thì k có sản phẩm để tư vấn
# Loại bỏ các kí tự lạ và đặc biệt
# Thực hiện nối theo kiểu semantic
# xử lí xong sẽ xử lí thêm cả việc loại bỏ một số đường link bằng bằng cách ctrl + F tìm //, xóa cụm câu:" Danh sách cửa hàng trải nghiệm sản phẩm Huawei."
# Xoas các câu bắt đầu bằng >>>>,>>,>>>,>,<

def clean_text(text):
    """Làm sạch text và đảm bảo kết thúc bằng dấu chấm."""
    # Escape & định dạng
    text = text.replace('\\"', '"')
    text = text.replace("\\", "")  # bỏ backslash dư
    text = text.replace("©", "")
    text = text.replace("®", "")
    text = text.replace("™", "")
    text = text.replace("@", "")
    text = text.replace("•", " ")  # hoặc thay bằng dấu -
    text = text.replace("~", " khoảng ")  # hoặc xấp xỉ
    text = text.replace("&", " và ")
    text = re.sub(r"[\x08\n\r•\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().strip(";:")
    if not text.endswith("."):
        text += "."
    return text


def remove_quoted_sentences(text):
    # Tách đoạn văn thành các câu dựa trên dấu chấm, xuống dòng, chấm hỏi, chấm than,...
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)

    # Lọc các câu không bắt đầu bằng dấu >
    cleaned_sentences = [s for s in sentences if not re.match(r'^\s*>+', s.strip())]

    # Nối lại thành văn bản hoàn chỉnh
    return ' '.join(cleaned_sentences)


def summarize_product_info(product_info):
    if not isinstance(product_info, dict):
        return ""
    stop_phrases = [r"\(.*?xem chi tiết.*?\)", r"\(.*?tại đây.*?\)", r"xem chi tiết", r"tại đây", r"click.*?đây"]

    def clean(v):
        for p in stop_phrases:
            v = re.sub(p, "", v, flags=re.IGNORECASE)
        return clean_text(v)

    return " ".join([clean(v) for v in product_info.values() if isinstance(v, str)])


def summarize_bonus_detailed(bonus_detailed):
    if not isinstance(bonus_detailed, dict):
        return ""
    ignore = "Xem chính sách ưu đãi dành cho thành viên Smember"
    return " ".join([
        clean_text(info["details"]) for info in bonus_detailed.values()
        if isinstance(info, dict) and info.get("details") != ignore
    ])


def summarize_promotion(promotion):
    if not isinstance(promotion, dict):
        return ""
    ignore = ["Không có khuyến mãi nào cho sản phẩm này", "Thông báo: Không có khuyến mãi nào cho sản phẩm này"]
    return " ".join([
        clean_text(v) for v in promotion.values()
        if isinstance(v, str) and v not in ignore
    ])


def summarize_warranty(warranty):
    if not isinstance(warranty, dict):
        return ""
    ignore = ["Không có bảo hành nào dành sản phẩm này", "Thông báo: Không có bảo hành nào dành sản phẩm này"]
    return " ".join([
        clean_text(v) for v in warranty.values()
        if isinstance(v, str) and v not in ignore
    ])


def summarize_technical_info(technical_info):
    if not isinstance(technical_info, dict):
        return ""
    ignore = "Không có thông tin về thông số kỹ thuật"

    def reclean(v):
        v = v.replace('\\"', '"')
        v = v.replace("\\", "")  # bỏ backslash dư
        v = v.replace("©", "")
        v = v.replace("®", "")
        v = v.replace("™", "")
        v = v.replace("@", "")
        v = v.replace("•", " ")  # hoặc thay bằng dấu -
        v = v.replace("~", " khoảng ")  # hoặc xấp xỉ
        v = v.replace("&", " và ")
        v = v.replace("\n", ", ")  # ✅ thay \n bằng dấu phẩy
        v = re.sub(r"[\x08\r•\-]+", " ", v)  # bỏ ký tự đặc biệt khác
        v = re.sub(r"\s+", " ", v)  # chuẩn hóa khoảng trắng
        v = v.strip().strip(";:,")
        if not v.endswith("."):
            v += "."
        return v

    return " ".join([remove_quoted_sentences(reclean(v)) for v in technical_info.values() if
                     isinstance(v, str) and v not in ignore])


def summarize_key_features(key_features):
    if not isinstance(key_features, dict):
        return ""
    return " ".join([
        clean_text(v) for v in key_features.values()
        if isinstance(v, str)
    ])


def summarize_main_content(main_content):
    if not isinstance(main_content, list):
        return ""
    chunks = []
    for section in main_content:
        if isinstance(section, dict) and "content" in section:
            for paragraph in section["content"]:
                if isinstance(paragraph, str):
                    cleaned = remove_quoted_sentences(clean_text(paragraph))
                    if cleaned:
                        chunks.append(cleaned)
    return " ".join(chunks).strip()


### ---- Xử lý toàn bộ file ----

json_data = [
    "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\raw_data_in_json\\smartphones.json",
    "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\raw_data_in_json\\tablets.json",
    "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\raw_data_in_json\\laptop.json",
    "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\raw_data_in_json\\earphone.json",
    "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\raw_data_in_json\\speakers.json",
    "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\raw_data_in_json\\watches.json"
]

json_formatted_data = [
    "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\final_data_in_json\\formatted_smartphones.json",
    "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\final_data_in_json\\formatted_tablets.json",
    "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\final_data_in_json\\formatted_laptop.json",
    "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\final_data_in_json\\formatted_earphone.json",
    "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\final_data_in_json\\formatted_speakers.json",
    "C:\\Users\\dangn\\PycharmProjects\\Law\\Đồ_Án\\crawldata\\data_in_json\\final_data_in_json\\formatted_watches.json",
]

for index, json_link in enumerate(json_data):
    with open(json_link, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    list_product_no_replicate = []

    for item in data:
        name = item.get("product_name")
        if isinstance(name, str) and name == "Không có tên sản phẩm":
            continue

        if name not in list_product_no_replicate:
            list_product_no_replicate.append(name)
        else:
            continue

        address = item.get("address")
        if (
                isinstance(address, dict)
                and address.get("Thông báo") == "Không có chi nhánh nào có sản phẩm này"
        ):
            continue

        new_item = {
            "product_name": item.get("product_name", "")
        }
        full_promotion = []

        if "product_info" in item:
            new_item["product_info_summary"] = summarize_product_info(item["product_info"])

        if "bonus_detailed" in item:
            full_promotion.append(summarize_bonus_detailed(item["bonus_detailed"]))

        if "promotion" in item:
            full_promotion.append(summarize_promotion(item["promotion"]))

        if "warranty" in item:
            new_item["warranty_summary"] = summarize_warranty(item["warranty"])

        if "technical_info" in item:
            new_item["technical_summary"] = summarize_technical_info(item["technical_info"])

        if "key_features" in item:
            new_item["feature_summary"] = summarize_key_features(item["key_features"])

        if "main_content" in item:
            new_item["content_summary"] = summarize_main_content(item["main_content"])

        # Gộp full_promtion
        new_item["full_promotion"] = " ".join(full_promotion)

        processed_data.append(new_item)

    # Ghi ra file mới
    output_path = json_formatted_data[index]
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(processed_data, file, ensure_ascii=False, indent=4)

    print(f"✅ Đã xử lý xong và lưu vào '{output_path}'")
