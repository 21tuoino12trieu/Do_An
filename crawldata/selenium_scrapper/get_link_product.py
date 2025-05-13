from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import concurrent.futures  # Để chạy song song
import time
import json


# Hàm cuộn đến phần tử
def scroll_into_view(driver, element):
    driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
    time.sleep(1)


# Hàm xử lý từng trang sản phẩm
def scrape_product_links(brand_link):
    driver = webdriver.Chrome()
    driver.maximize_window()
    driver.get(brand_link)

    print(f"🔄 Đang lấy sản phẩm từ thương hiệu: {brand_link}")

    # Chờ trang tải xong
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    # **Nhấn "Xem thêm sản phẩm" cho đến khi hết**
    while True:
        try:
            button_show_more_product = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".button__show-more-product"))
            )
            if button_show_more_product.is_displayed() and button_show_more_product.is_enabled():
                scroll_into_view(driver, button_show_more_product)
                driver.execute_script("arguments[0].click();", button_show_more_product)
                print("✅ Đã click 'Xem thêm sản phẩm'")
                WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".product-item")))
            else:
                break
        except:
            print("❌ Không còn nút 'Xem thêm sản phẩm'")
            break

    # **Lấy danh sách sản phẩm**
    product_elements = driver.find_elements(By.CSS_SELECTOR, ".product-item a")
    product_links = [product.get_attribute("href") for product in product_elements if product.get_attribute("href")]
    # **Duyệt từng sản phẩm**
    for product_link in product_links:
        try:
            print(f"📌 Đang mở sản phẩm: {product_link}")

            # Mở tab mới
            driver.execute_script("window.open(arguments[0]);", product_link)
            driver.switch_to.window(driver.window_handles[1])  # Chuyển sang tab mới

            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            # **Tìm sản phẩm con (dung lượng khác)**
            list_links_capacity = driver.find_elements(By.CSS_SELECTOR, ".box-linked .list-linked a")

            if list_links_capacity:
                # Lấy danh sách sản phẩm con nếu có
                product_links_capacity = [link.get_attribute("href") for link in list_links_capacity]
                all_product_links.extend(product_links_capacity)
                print(f"🔹 Đã tìm thấy {len(product_links_capacity)} sản phẩm con.")
            else:
                # Nếu không có sản phẩm con, lưu sản phẩm gốc
                all_product_links.append(product_link)

            # Đóng tab sản phẩm & quay lại danh sách
            driver.close()
            driver.switch_to.window(driver.window_handles[0])  # Quay lại tab cũ

            # **ĐÓNG POP UP HIỆN LÊN ĐĂNG KÍ MEMBER**
            try:
                close_button = WebDriverWait(driver, 1).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, ".cancel-button-top"))
                )
                close_button.click()
                print("✅ Đã đóng form đăng ký!")
            except:
                print("⚠ Không tìm thấy form đăng ký hoặc không thể đóng.")

        except Exception as e:
            print(f"❌ Lỗi khi lấy sản phẩm {product_link}: {e}")
            driver.close()
            driver.switch_to.window(driver.window_handles[0])  # Đảm bảo về lại tab chính
    print(f"✅ Đã lấy {len(all_product_links)} sản phẩm từ {brand_link}")
    driver.quit()
    return product_links


# **Khởi động trình duyệt để lấy danh sách thương hiệu**
driver = webdriver.Chrome()
driver.maximize_window()
homepage_url = "https://cellphones.com.vn/do-choi-cong-nghe/dong-ho-thong-minh-nghe-goi.html"
driver.get(homepage_url)

WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".brands__content")))

brand_elements = driver.find_elements(By.CSS_SELECTOR, ".brands__content .list-brand__item")
brand_links = [brand.get_attribute("href") for brand in brand_elements]

driver.quit()  # Đóng trình duyệt chính để tránh lãng phí tài nguyên

# **Chạy song song việc lấy dữ liệu từ các thương hiệu**
all_product_links = []

with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
    futures = {executor.submit(scrape_product_links, link): link for link in brand_links}

    for future in concurrent.futures.as_completed(futures):
        try:
            product_links = future.result()

            all_product_links.extend(product_links)
        except Exception as e:
            print(f"❌ Lỗi khi lấy sản phẩm từ một thương hiệu: {e}")

# **Lưu vào file JSON**
with open("C:\\Users\\dangn\\PycharmProjects\\Law\\crawldata\\multi_threads_watch_product_links.json", "w", encoding="utf-8") as f:
    json.dump(all_product_links, f, indent=4, ensure_ascii=False)

print(f"🎉 Hoàn thành! Đã thu thập {len(all_product_links)} sản phẩm.")
