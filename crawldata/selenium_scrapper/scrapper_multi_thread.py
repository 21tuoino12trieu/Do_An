from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import concurrent.futures  # Để chạy đa luồng
import time
import json
import multiprocessing  # Sử dụng Manager để lưu dữ liệu giữa các tiến trình

# Hàm cuộn đến phần tử
def scroll_into_view(driver, element):
    driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
    time.sleep(1)  # Chờ trang ổn định sau khi cuộn

def scrape_page(url):
    # Khởi động trình duyệt
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Chạy chế độ headless để không mở cửa sổ
    driver = webdriver.Chrome(options=options)

    # Truy cập trang web
    driver.get(url)

    # Lấy chiều cao trang và cuộn xuống từng phần
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Tên sản phẩm

    product_name_selectors = ["div.box-product-name", ".boxInfoRight .product-title"]
    product_name = None

    try:
        for selector in product_name_selectors:
            product_name_elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if product_name_elements:
                product_name = product_name_elements[0].text.strip()
                print(f"Đã tìm thấy tên sản phẩm: {product_name}")
                break  # Dừng vòng lặp nếu đã tìm thấy tên sản phẩm
    except:
        print("Lỗi khi lấy tên sản phẩm.")

    # Nếu lấy tên sản phẩm từ ".boxInfoRight .product-title", cần ghép thêm dung lượng
    if product_name and ".boxInfoRight .product-title" in selector:
        try:
            capacity = driver.find_element(By.CSS_SELECTOR,
                                           ".boxInfoRight .product-tab-child .product-item.is-active").text.strip()
            product_name = f"{product_name} {capacity}"
            print(f"Đã cập nhật tên sản phẩm kèm dung lượng: {product_name}")
        except:
            print("Không tìm thấy dung lượng sản phẩm.")

    # Giá sản phẩm
    # Giá tiền của sản phẩm
    product_price = None

    try:
        list_price_css_selectors = [".box-detail-product .product__price--show", ".tpt-boxs .tpt---sale-price",
                                    ".boxInfoRight .product__price--show"]
        for price_css_selector in list_price_css_selectors:
            elements = driver.find_elements(By.CSS_SELECTOR, price_css_selector)
            if elements:
                if len(elements) == 2:
                    product_price = {
                        "Giá khi thu cũ lên đời": elements[0].text.splitlines()[0],
                        "Giá chính thức": elements[1].text,
                    }
                    print("Đã tìm thấy giá sản phẩm")
                elif len(elements) == 1:
                    product_price = {
                        "Giá chính thức": elements[0].text,
                    }
                    print("Đã tìm thấy giá sản phẩm")
                break  # Nếu đã lấy được giá, không cần kiểm tra tiếp
    except:
        product_price["Thông báo"] = "⚠ Không thể tìm thấy giá sản phẩm"

    # **THÔNG TIN SẢN PHẨM**
    product_info = {}
    try:
        product_info_elements = driver.find_elements(By.CSS_SELECTOR, ".item-warranty-info")
        if product_info_elements:
            for index, element in enumerate(product_info_elements):
                scroll_into_view(driver, element)
                product_info[f"Thông tin {index + 1}"] = element.text.strip()
            print("Đã tìm thấy thông tin sản phẩm")
        else:
            product_info["Thông báo"] = "Không có thông tin về sản phẩm này"
    except:
        product_info["Lỗi"] = "Không thể lấy thông tin sản phẩm"

    # **ƯU ĐÃI RIÊNG CHO MEMBER**
    preference = {}
    try:
        preference_elements = driver.find_elements(By.CSS_SELECTOR, ".exclusive-price-block p")

        if preference_elements:
            for index, element in enumerate(preference_elements):
                scroll_into_view(driver, element)  # Cuộn chuột đến từng phần tử
                preference[f"Ưu đãi riêng {index + 1}"] = element.text.strip()
            print("Đã tìm thấy ưu đãi riêng cho member")
        else:
            preference["Thông báo"] = "Không có ưu đãi riêng nào dành sản phẩm này"
    except:
        preference["Lỗi"] = "Không lấy được ưu đãi nào"

    # **KHUYẾN MÃI**
    promotion = {}
    try:
        promotion_elements = driver.find_elements(By.CSS_SELECTOR, ".box-product-promotion-detail")

        if promotion_elements:
            for index, element in enumerate(promotion_elements):
                scroll_into_view(driver, element)  # Cuộn chuột đến từng phần tử
                promotion[f"Khuyến mãi thứ {index + 1}"] = element.text.strip()
            print("Đã tìm thấy khuyến mãi cho sản phẩm này")
        else:
            promotion["Thông báo"] = "Không có khuyến mãi nào cho sản phẩm này"
    except:
        promotion["Lỗi"] = "Không lấy được khuyến mãi nào"

    # **ƯU ĐÃI THÊM**
    bonus_detailed = {}
    try:
        bonus_elements = driver.find_elements(By.CSS_SELECTOR, ".render-promotion li")

        if bonus_elements:
            for index, bonus in enumerate(bonus_elements):
                scroll_into_view(driver, bonus)  # Cuộn chuột đến từng phần tử
                link_elements = bonus.find_elements(By.CSS_SELECTOR, "a")

                url = link_elements[0].get_attribute("href") if link_elements else "Không bóc được giá trị"
                details = link_elements[0].text.strip() if link_elements else "Không bóc được giá trị"

                bonus_detailed[f"Ưu đãi thứ {index + 1}"] = {"url": url, "details": details}
            print("Đã tìm thấy ưu đãi thêm cho sản phẩm này")
        else:
            bonus_detailed["Thông báo"] = "Không có ưu đãi thêm nào dành sản phẩm này"
    except:
        bonus_detailed["Lỗi"] = "Không lấy được ưu đãi thêm nào"

    # **CHÍNH SÁCH BẢO HÀNH**
    try:
        warranty_section = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#extendedWarranty"))
        )
        scroll_into_view(driver, warranty_section)  # Cuộn chuột đến phần bảo hành
    except:
        print("Không tìm thấy mục bảo hành.")

    warranty = {}
    try:
        warranty_elements = driver.find_elements(By.CSS_SELECTOR, "#extendedWarranty label")

        if warranty_elements:
            for index, element in enumerate(warranty_elements):
                scroll_into_view(driver, element)  # Cuộn chuột đến từng chính sách bảo hành
                text_content = element.text.split("\n")[0].strip()
                try:
                    price = element.find_element(By.CSS_SELECTOR, ".price").text.strip()
                except:
                    price = "Không có giá"
                warranty[f"Chính sách thứ {index + 1}"] = f"{text_content} - Giá: {price}"
            print("Đã tìm chính sách bảo hành cho sản phẩm")
        else:
            warranty["Thông báo"] = "Không có bảo hành nào dành sản phẩm này"
    except:
        warranty["Lỗi"] = "Không lấy được ra chính sách bảo hành nào"

    # Ẩn iframe live chat để tránh lỗi click bị chặn
    try:
        chat_iframe = driver.find_element(By.ID, "cs_chat_iframe")
        driver.execute_script("arguments[0].style.display = 'none';", chat_iframe)
        print("Đã ẩn iframe live chat!")
    except:
        print("Không tìm thấy iframe live chat.")

    # **THÔNG SỐ KỸ THUẬT SẢN PHẨM**
    # Click vào nút thông số kỹ thuật
    try:
        tech_button = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".button__show-modal-technical"))
        )
        scroll_into_view(driver, tech_button)  # Cuộn đến nút trước khi click
        driver.execute_script("arguments[0].click();", tech_button)
    except:
        print("Không tìm thấy hoặc không thể click vào nút hiển thị thông số kỹ thuật.")

    # Lấy thông số kỹ thuật
    technical_info = {"Thông báo": "Không có thông tin về thông số kỹ thuật"}

    try:
        categories = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".technical-content-modal-item"))
        )
        technical_info = {}
        for category in categories:
            title = category.find_element(By.CSS_SELECTOR, "p.title.is-6.m-2").text.strip()
            specs = category.find_elements(By.CSS_SELECTOR,
                                           ".px-3.py-2.is-flex.is-align-items-center.is-justify-content-space-between")

            details = []
            for spec in specs:
                try:
                    key = spec.find_element(By.TAG_NAME, "p").text.strip()
                    value = spec.find_element(By.TAG_NAME, "div").text.strip()
                    details.append(f"{key}: {value}")
                except:
                    continue  # Bỏ qua nếu không thể lấy key-value

            technical_info[title] = "\n".join(details)
        print("Đã tìm thấy thông số kỹ thuật cho sản phẩm")
    except:
        print("Không thể lấy thông tin kỹ thuật.")

    # Đóng modal thông số kỹ thuật nếu có
    try:
        close_button = driver.find_element(By.CSS_SELECTOR, ".close-button-modal")
        scroll_into_view(driver, close_button)  # Cuộn đến nút trước khi click
        close_button.click()
    except:
        print("Không tìm thấy nút đóng modal.")

    # **ĐÓNG POP UP HIỆN LÊN ĐĂNG KÍ MEMBER**
    try:
        close_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".cancel-button-top"))
        )
        close_button.click()
        print("Đã đóng form đăng ký!")
    except:
        print("Không tìm thấy form đăng ký hoặc không thể đóng.")

    # ** CÁC CHI NHÁNH CỬA HÀNG HIỆN CÓ SẢN PHẨM**
    # Click vào nút chọn tỉnh/thành phố
    try:
        button_branch = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".button__change-province"))
        )
        scroll_into_view(driver, button_branch)  # Cuộn đến nút trước khi click
        button_branch.click()
    except:
        print("Không thể click vào nút thay đổi tỉnh/thành phố.")

    # Chọn Hà Nội
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#popUpChangeProvince.is-active"))
        )
        all_provinces = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".modal__button"))
        )
        for province in all_provinces:
            if "Hà Nội" in province.text.strip():
                scroll_into_view(driver, province)  # Cuộn đến tỉnh trước khi click
                province.click()
                print("Đã chọn Hà Nội!")
                break
    except:
        print("Không tìm thấy hoặc không thể chọn Hà Nội.")

    # Chờ trang cập nhật sau khi chọn tỉnh
    time.sleep(10)
    # scroll_into_view(driver, branch_elements)

    # Lấy danh sách chi nhánh có hàng
    branch = {}

    try:
        branch_elements = driver.find_elements(By.CSS_SELECTOR, ".address")
        if branch_elements:
            branch = {}
            for index, element in enumerate(branch_elements):
                address = element.get_attribute("title")
                scroll_into_view(driver, element)
                branch[f"Chi nhánh {index + 1}"] = address
            print("Đã tìm thấy chi nhánh có sản phẩm")
        else:
            branch["Thông báo"] = "Không có chi nhánh nào có sản phẩm này"
    except:
        branch["Lỗi"] = "Không thể lấy danh sách chi nhánh"

    # ** ĐẶC ĐIỂM NỔI BẬT CỦA SẢN PHẨM **
    cpsContent = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#cpsContent"))
    )
    scroll_into_view(driver, cpsContent)

    key_features = {}
    try:
        key_features_elements = driver.find_elements(By.CSS_SELECTOR, ".ksp-content li")
        if key_features_elements:
            for index, element in enumerate(key_features_elements):
                feature = element.text.strip()
                scroll_into_view(driver, element)
                key_features[f"Đặc điểm {index + 1}"] = feature
            print("Đã tìm thấy đặc điểm nổi bất của sản phẩm")
        else:
            key_features["Thông báo"] = "Không có thông tin nổi bật về sản phẩm này"
    except:
        key_features["Lỗi"] = "Không có thông tin nổi bật của sản phẩm"

    # ** BÓC TÁCH NỘI DUNG CHÍNH BÀI VIẾT **
    try:
        show_more_button = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn-show-more.button__content-show-more"))
        )
        driver.execute_script("arguments[0].click();", show_more_button)
        print("Đã mở rộng nội dung!")
        time.sleep(2)
    except:
        print("Không có nút 'Xem thêm' hoặc đã được hiển thị đầy đủ.")

    # Lấy tất cả phần tử trong nội dung chính (h1, h2, h3, p)
    grouped_content = []
    try:
        content_section = driver.find_element(By.ID, "cpsContentSEO")
        main_content_elements = content_section.find_elements(By.CSS_SELECTOR, "h1, h2, h3, p")

        current_section = None
        if main_content_elements:
            for element in main_content_elements:
                scroll_into_view(driver, element)
                tag_name = element.tag_name.lower()
                text = element.text.strip()

                if text:
                    if tag_name in ["h1", "h2", "h3"]:  # Nếu là tiêu đề mới
                        current_section = {
                            "title": text,
                            "content": []
                        }
                        grouped_content.append(current_section)
                    elif tag_name == "p" and current_section:
                        current_section["content"].append(text)
            print("Đã tìm thấy nội dung bài viết mở rộng của sản phẩm")
        else:
            grouped_content["Thông báo"] = "Không có bài viết mở rộng về sản phẩm này"
    except:
        print("Không thể lấy nội dung bài viết.")

    # Lưu dữ liệu vào JSON
    exacted_data = {
        "product_name": product_name if product_name else "Không có tên sản phẩm",
        "price": product_price if product_price else {"Thông báo": "Không có giá"},
        "product_info": product_info if product_info else {"Thông báo": "Không có thông tin"},
        "preference": preference if preference else {"Thông báo": "Không có ưu đãi"},
        "bonus_detailed": bonus_detailed if bonus_detailed else {"Thông báo": "Không có ưu đãi thêm"},
        "promotion": promotion if promotion else {"Thông báo": "Không có khuyến mãi"},
        "warranty": warranty if warranty else {"Thông báo": "Không có bảo hành"},
        "address": branch if branch else {"Thông báo": "Không có chi nhánh"},
        "technical_info": technical_info if technical_info else {"Thông báo": "Không có thông số kỹ thuật"},
        "key_features": key_features if key_features else {"Thông báo": "Không có đặc điểm nổi bật"},
        "main_content": grouped_content if grouped_content else [{"title": "Không có nội dung", "content": []}],
    }

    # Kết thúc trình duyệt
    driver.quit()
    return exacted_data


if __name__ == "__main__":
    # Load danh sách các link sản phẩm
    with open("C:\\Users\\dangn\\PycharmProjects\\Law\\crawldata\\multi_threads_tablet_product_links.json", "r", encoding="utf-8") as f:
        smartphone_product_links = json.load(f)
    # Tạo danh sách để chứa dữ liệu
    max_workers = 10
    all_data = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(scrape_page, url) for url in smartphone_product_links]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    all_data.append(result)  # Thêm dữ liệu vào danh sách
            except Exception as e:
                print(f"❌ Lỗi khi xử lý một sản phẩm: {e}")

    # Ghi dữ liệu vào file JSON
    with open("C:\\Users\\dangn\\PycharmProjects\\Law\\crawldata\\tablets.json", "w", encoding="utf-8") as f:
        json.dump(list(all_data), f, indent=4, ensure_ascii=False)

    print(f"🎉 Hoàn thành! Đã thu thập {len(all_data)} sản phẩm.")