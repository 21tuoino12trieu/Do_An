"""
File chứa tất cả các prompt cho hệ thống RAG cải tiến
"""

# Prompt phân loại câu hỏi xem có đang liên quan hay không
QUERY_DOMAIN_CLASSIFICATION_PROMPT = """
Với vai trò là một bộ phân loại truy vấn tự động, nhiệm vụ của bạn là xác định miền chủ đề của câu hỏi được cung cấp và phân loại nó vào một trong hai nhóm được định nghĩa.

Truy vấn cần phân loại:
"{clarified_query}"

Hãy phân tích truy vấn này và gán nhãn phân loại chính xác dựa trên các quy tắc sau:

**Nhãn Phân loại:**
-   **RELATED:** Gán nhãn này nếu truy vấn liên quan đến **thiết bị di động hoặc thiết bị điện tử tiêu dùng**. Các chủ đề bao gồm (nhưng không giới hạn): smartphone, tablet, laptop, tai nghe, loa, đồng hồ thông minh, thông số kỹ thuật, chức năng, hiệu năng, giá bán, địa điểm mua, so sánh giữa các thiết bị, hướng dẫn sử dụng, khắc phục sự cố liên quan đến phần cứng/phần mềm của các thiết bị này.
-   **UNRELATED:** Gán nhãn này nếu truy vấn thuộc **bất kỳ chủ đề nào khác** không liên quan đến thiết bị di động hoặc thiết bị điện tử tiêu dùng như định nghĩa ở trên. Các chủ đề UNRELATED bao gồm (nhưng không giới hạn): thời tiết, thể thao, lịch sử, chính trị, tin tức chung, y tế, giáo dục, văn hóa, nghệ thuật, ẩm thực, du lịch, ...

**Định dạng Đầu ra Bắt buộc:**
Chỉ trả về **duy nhất một từ khóa** là nhãn phân loại đã chọn (`RELATED` hoặc `UNRELATED`). Tuyệt đối không thêm bất kỳ ký tự, dấu câu, giải thích, hoặc văn bản nào khác trước, sau, hoặc cùng với từ khóa phân loại.
"""

# Prompt này dùng để xử lí các câu hỏi không liên quan đến chủ đề:
QUERY_UNRELATED_HANDLE_PROMPT = """
Bạn là một trợ lý tư vấn thiết bị di động chuyên nghiệp, thân thiện và am hiểu rộng, đặc biệt có khả năng xử lý khéo léo các câu hỏi không thuộc chuyên môn chính (thiết bị di động/điện tử).
Bạn nhận được một câu hỏi đã được xác định là KHÔNG liên quan trực tiếp đến thiết bị di động:
Câu hỏi gốc: "{clarified_query}"

Nhiệm vụ của bạn là:
1.  **Trả lời Ngắn gọn & Thân thiện:** Cung cấp một câu trả lời trực tiếp, ngắn gọn và hữu ích nhất có thể cho câu hỏi gốc của người dùng. Câu trả lời này chỉ nên gói gọn trong **tối đa 2 câu**.
2.  **Chuyển hướng Sang Trọng & Tự Nhiên:** Ngay sau câu trả lời ngắn gọn, hãy tìm cách chuyển hướng cuộc hội thoại một cách mượt mà và hấp dẫn sang chủ đề thiết bị di động hoặc công nghệ liên quan. Việc chuyển hướng này phải cảm thấy tự nhiên, không gượng ép, và gợi mở sự tò mò của người dùng về lĩnh vực chuyên môn của bạn. Hãy dựa vào chủ đề của câu hỏi gốc để tạo câu chuyển hướng phù hợp.

        *Gợi ý cách chuyển hướng dựa trên chủ đề (Hãy sáng tạo thêm nếu chủ đề khác):*
        * **Nếu hỏi về Thời tiết:** "...À, nói về công nghệ tiện ích, bạn có biết các ứng dụng dự báo thời tiết trên smartphone giờ xịn lắm không? Giao diện trực quan mà thông tin lại siêu chính xác đấy!"
        * **Nếu hỏi về Thể thao:** "...Tiện đây, bạn có hay theo dõi hoạt động không? Các dòng smartwatch mới giờ có thể theo dõi chi tiết mọi chỉ số tập luyện của bạn đấy, rất hay!"
        * **Nếu hỏi về Nấu ăn/Ẩm thực:** "...Nhân tiện, bạn có biết nhiều máy tính bảng màn hình lớn cực kỳ hữu ích cho việc nấu ăn không? Dễ dàng xem công thức khi thực hành luôn."
        * **Nếu hỏi về Du lịch:** "...Nói đến đi chơi, một chiếc điện thoại pin 'trâu' hay tai nghe chống ồn tốt là không thể thiếu đúng không? Giúp chuyến đi thoải mái hơn nhiều."
3.  **Kết thúc Gợi mở:** Kết thúc toàn bộ phản hồi bằng một câu hỏi mở hoặc lời mời trực tiếp, khuyến khích người dùng tương tác tiếp về chủ đề thiết bị di động. Mục tiêu là tạo cơ hội để bạn bắt đầu tư vấn về chuyên môn.
        *Ví dụ câu kết thúc:*
        * "Bạn có đang quan tâm đến thiết bị công nghệ nào gần đây không? Tôi có thể tư vấn giúp bạn đấy."
        * "Có thiết bị điện tử nào mà bạn đang tò mò muốn tìm hiểu thêm không nhỉ?"
        * "Nếu bạn cần tìm một chiếc [Điện thoại/Laptop/Tai nghe/...] phù hợp với nhu cầu của mình, cứ hỏi tôi nhé!"

     **Ngữ điệu & Phong cách:** Luôn giữ sự nhiệt tình, gần gũi, hữu ích và chuyên nghiệp trong suốt phản hồi. Lời nói phải tự nhiên như đang trò chuyện, không giống một kịch bản khô cứng.

    **Cấu trúc Đầu ra Bắt buộc:**
    Kết quả bạn trả về PHẢI là một đoạn văn bản liền mạch, bao gồm:
    [Câu trả lời ngắn gọn câu hỏi gốc (tối đa 2 câu)] + [Câu/đoạn chuyển hướng tự nhiên] + [Câu kết thúc gợi mở/lời mời].
"""

# Prompt để trích xuất nhiều tên sản phẩm từ câu hỏi so sánh
EXTRACT_MULTIPLE_PRODUCTS_PROMPT = """
Từ câu hỏi so sánh sau, xác định chính xác tên các sản phẩm được so sánh:

"{clarified_query}"

1. Xác định tên đầy đủ của từng sản phẩm, bao gồm cả thương hiệu, dòng sản phẩm, phiên bản
2. Loại bỏ các từ không phải là một phần của tên sản phẩm
3. Phân biệt rõ ràng các sản phẩm riêng biệt

Ví dụ:
- Từ "so sánh iPhone 13 Pro Max và Samsung Galaxy S22 Ultra" → ["iPhone 13 Pro Max", "Samsung Galaxy S22 Ultra"]
- Từ "iPhone 14 tốt hơn Xiaomi 13 không?" → ["iPhone 14", "Xiaomi 13"]

Trả về danh sách các sản phẩm định dạng JSON, chỉ các tên sản phẩm:
"""

# Prompt để xác định các trường cần so sánh
IDENTIFY_COMPARISON_FIELDS_PROMPT = """
Phân tích câu hỏi so sánh sau và xác định các trường thông tin cần so sánh:

"{clarified_query}"

Các trường có sẵn:
- product_info: Thông tin chung về sản phẩm
- warranty: Thông tin bảo hành
- technical: Thông số kỹ thuật
- feature: Tính năng nổi bật
- content: Nội dung chi tiết về sản phẩm
- full_promotion: Thông tin khuyến mãi

Phân tích nội dung câu hỏi và xác định những trường thông tin liên quan để so sánh. 
Nếu câu hỏi chỉ rõ khía cạnh so sánh (ví dụ: camera, pin, hiệu năng), hãy chọn các trường liên quan.
Nếu câu hỏi chung chung, hãy chọn các trường phù hợp nhất.

Trả về kết quả theo định dạng trường1,trường2,trường3. Không thêm giải thích hoặc văn bản khác:
"""

# Prompt để tóm tắt thông số kỹ thuật
SUMMARIZE_TECHNICAL_PROMPT = """
Từ thông số kỹ thuật sau, rút ra 3-5 chỉ số quan trọng nhất dưới dạng danh sách ngắn gọn:

{full_info}

Kết quả:
"""

# Prompt để tóm tắt tính năng nổi bật
SUMMARIZE_FEATURE_PROMPT = """
Từ các tính năng sau, liệt kê 2-3 tính năng nổi bật nhất, ngắn gọn:

{full_info}

Kết quả:
"""

# Prompt để tóm tắt thông tin chung
SUMMARIZE_GENERAL_INFO_PROMPT = """
Tóm tắt thông tin sau thành 2-3 điểm chính, ngắn gọn:

{full_info}

Kết quả:
"""

# Prompt để tạo câu trả lời cho câu hỏi so sánh
COMPARISON_RESPONSE_PROMPT = """
Tạo câu trả lời so sánh các sản phẩm:

Câu hỏi gốc: "{original_query}"
Câu hỏi đã làm rõ: "{clarified_query}"
Sản phẩm so sánh: {products}
Các tiêu chí so sánh: {fields}

Dữ liệu so sánh:
{comparison_data}

Hãy tạo một câu trả lời chi tiết so sánh các sản phẩm theo các tiêu chí đã nêu. Câu trả lời cần:

1. Mở đầu bằng việc xác nhận các sản phẩm được so sánh
2. So sánh chi tiết từng sản phẩm theo từng tiêu chí, nêu rõ điểm mạnh/yếu
3. Đưa ra đánh giá khách quan về ưu/nhược điểm của từng sản phẩm
4. Nêu rõ sản phẩm nào phù hợp cho những nhu cầu cụ thể
5. Kết luận với lời khuyên tổng quan

Giữ câu trả lời khách quan, không thiên vị, dễ hiểu và hữu ích cho người dùng.
"""

# giới thiệu cho tôi về Tai nghe Bluetooth chụp tai OneOdio A10

# Prompt làm rõ câu hỏi người dùng
QUERY_CLARIFICATION_PROMPT = """
Với vai trò là bộ xử lý và làm rõ truy vấn chuyên biệt cho hệ thống Tìm kiếm Tăng cường (RAG), nhiệm vụ của bạn là phân tích truy vấn thô của người dùng và biến nó thành một truy vấn tối ưu nhất để hệ thống có thể tìm kiếm các tài liệu liên quan một cách chính xác.

Truy vấn gốc cần xử lý:
"{user_query}"

Hãy thực hiện quy trình sau để làm rõ và tái cấu trúc truy vấn:
1.  **Xác định Ý định Cốt lõi:** Phân tích truy vấn gốc để hiểu rõ mục đích chính và nhu cầu thông tin cụ thể của người dùng. Họ thực sự muốn tìm kiếm điều gì?
2.  **Trích xuất & Mở rộng Khái niệm:** Xác định các khái niệm, thực thể, thuật ngữ, từ khóa quan trọng nhất trong truy vấn. Nếu có từ viết tắt, biệt ngữ, hoặc thuật ngữ không rõ ràng, hãy giải thích hoặc mở rộng chúng thành dạng đầy đủ (nếu ngữ cảnh cho phép và không suy diễn).
3.  **Tái Cấu trúc Truy vấn:** Dựa trên ý định đã làm rõ và các khái niệm/từ khóa đã trích xuất, hãy viết lại truy vấn gốc thành một câu hỏi hoặc cụm từ tìm kiếm duy nhất. Truy vấn mới này phải rõ ràng, cụ thể hơn, đầy đủ ngữ nghĩa cần thiết cho việc tìm kiếm, và loại bỏ sự mơ hồ nếu có trong truy vấn gốc.

**Nguyên tắc quan trọng:**
- Chỉ làm rõ và tái cấu trúc dựa trên thông tin *có sẵn* trong truy vấn gốc.
- Tuyệt đối KHÔNG suy diễn, bịa thêm, hoặc thêm bất kỳ thông tin, bối cảnh mới nào không được đề cập trực tiếp trong "{user_query}".
- Truy vấn cuối cùng phải tập trung vào việc tối đa hóa khả năng tìm thấy tài liệu phù hợp trong cơ sở tri thức.

**Đầu ra mong muốn:**
Chỉ trả về *duy nhất* truy vấn đã được xử lý và làm rõ (kết quả của bước 3). Không bao gồm các bước phân tích trung gian hay bất kỳ văn bản giải thích nào khác.
"""

# Prompt phân loại câu hỏi
QUERY_CLASSIFICATION_PROMPT = """
Phân loại câu hỏi sau:

"{clarified_query}"

Đây là hệ thống tư vấn sản phẩm điện tử với 2 loại câu hỏi chính:

1. GENERAL: Câu hỏi chung chung về một loại sản phẩm, người dùng muốn gợi ý nhiều sản phẩm
   Ví dụ: "Tư vấn tai nghe có thiết kế đẹp", "Điện thoại chụp ảnh tốt", "Laptop cho sinh viên", "Tư vấn cho tôi tai nghe chống ồn"

2. SPECIFIC: Câu hỏi về một sản phẩm cụ thể (cần xác định thêm loại thông tin)
   Ví dụ: "Thông số kỹ thuật của iPhone 13", "Giá của tai nghe SoundPeats T3", "Đánh giá Samsung Galaxy S22"

Nếu câu hỏi thuộc loại SPECIFIC, xác định thêm loại thông tin cần trả lời:
   a. VECTOR: Thông tin về toàn bộ sản phẩm, thông số kỹ thuật, chính sách bảo hành, chính sách khuyến mãi ưu đãi, đặc điểm nổi bật, đánh giá chung  
   b. SQL: Thông tin về giá cả, địa điểm bán, cửa hàng gần nhất, số lượng hàng còn lại. Bao gồm các câu hỏi như:
      - Về địa chỉ: "Cửa hàng nào gần nhất có bán iPhone 14?", "Ở đâu bán Samsung S22?"
      - Về giá: "Giá của Xiaomi Redmi Note 11 là bao nhiêu?", "Tai nghe AirPods Pro 2 giá thế nào?"
      - Về số lượng sản phẩm: "Có bao nhiêu điện thoại trong tầm giá 5 triệu?", "Liệt kê laptop dưới 15 triệu"
   c. HYBRID: Cần cả thông tin VECTOR và SQL

Trả về theo định dạng:
GENERAL hoặc 
SPECIFIC-VECTOR hoặc
SPECIFIC-SQL hoặc
SPECIFIC-HYBRID

Chỉ trả về định dạng trên, không thêm bất kỳ giải thích nào khác.
"""

# Prompt xác định field tối ưu cho câu hỏi chung chung
GENERAL_FIELD_IDENTIFICATION_PROMPT = """
Phân tích câu hỏi chung chung sau và xác định trường dữ liệu phù hợp nhất:

"{clarified_query}"

Các trường có sẵn:
- product_name: Tên sản phẩm
- product_info: Thông tin chung về sản phẩm
- warranty: Thông tin bảo hành, đổi trả
- technical: Thông số kỹ thuật, cấu hình
- feature: Tính năng nổi bật
- content: Nội dung chi tiết, đánh giá tổng quan
- full_promotion: Thông tin khuyến mãi

Phân tích:
1. Nếu người dùng cần thông tin về tên sản phẩm, mẫu mã, hãng sản xuất: product_name
2. Nếu người dùng cần thông tin về bảo hành, đổi trả: warranty
3. Nếu người dùng cần thông tin về thông số kỹ thuật, cấu hình: technical
4. Nếu người dùng cần thông tin về tính năng nổi bật: feature
5. Nếu người dùng cần thông tin đánh giá tổng quan, thiết kế: content
6. Nếu người dùng cần thông tin khuyến mãi: full_promotion
7. Nếu người dùng cần thông tin chung về sản phẩm: product_info

Chỉ trả về tên trường tối ưu nhất (chỉ một trường duy nhất) mà không có giải thích:
"""

# Prompt xác định tên sản phẩm từ câu hỏi cụ thể
EXTRACT_PRODUCT_NAME_PROMPT = """
Từ câu hỏi sau, xác định và trích xuất **duy nhất** tên sản phẩm mà người dùng đang hỏi:

"{clarified_query}"

Hãy thực hiện theo các bước sau để trích xuất **tên sản phẩm DUY NHẤT** từ truy vấn:
1.  Xác định tên sản phẩm điện tử đầy đủ nhất có thể được đề cập **duy nhất một lần** trong truy vấn, bao gồm cả thương hiệu, dòng sản phẩm, phiên bản, model nếu có.
2.  Tập trung vào các từ khóa là tên riêng của sản phẩm.
3.  Loại bỏ các từ khóa chung chung như "smartphone", "tai nghe", "laptop", "tivi", v.v., nếu không phải là một phần của tên sản phẩm cụ thể.

Ví dụ:
- Từ "tư vấn tai nghe SoundPeats Air3 Deluxe HS" → "SoundPeats Air3 Deluxe HS"
- Từ "thông số kỹ thuật iPhone 13 Pro Max" → "iPhone 13 Pro Max"
- Từ "Samsung Galaxy S22 Ultra mua ở đâu" → "Samsung Galaxy S22 Ultra"

**Yêu Cầu Đầu Ra BẮT BUỘC:**
Chỉ trả về **duy nhất tên sản phẩm** đã trích xuất. Không thêm bất kỳ văn bản giới thiệu, giải thích, dấu câu trang trí, hoặc ký tự nào khác.

Tên sản phẩm trích xuất:
"""

# Prompt xác định fields cần lấy thông tin cho câu hỏi cụ thể
SPECIFIC_FIELDS_IDENTIFICATION_PROMPT = """
Phân tích câu hỏi sau và xác định các trường thông tin cần lấy:

"{clarified_query}"

Các trường có sẵn:
- product_info: Thông tin chung về sản phẩm
- warranty: Thông tin bảo hành
- technical: Thông số kỹ thuật
- feature: Tính năng nổi bật
- content: Nội dung chi tiết về sản phẩm
- full_promotion: Thông tin khuyến mãi

Phân tích nội dung câu hỏi và xác định những trường thông tin liên quan trực tiếp. Chọn từ 1-3 trường thông tin phù hợp nhất với câu hỏi.

Nếu câu hỏi chung chung không chỉ rõ khía cạnh nào, trả về "all" để lấy tất cả các trường.

Trả về kết quả theo định dạng trường1,trường2,trường3 hoặc all. Không thêm giải thích hoặc văn bản khác:
"""

# Prompt tạo câu lệnh SQL
SQL_GENERATION_PROMPT = """
Tạo câu lệnh SQL cho câu hỏi sau:

"{clarified_query}"

Tìm thông tin liên quan đến sản phẩm: "{product_name}"

Bảng products có cấu trúc như sau:
- id: INTEGER PRIMARY KEY
- product_name: TEXT (tên sản phẩm)
- price: TEXT (giá sản phẩm, ví dụ: "5.990.000₫")
- address: TEXT (địa chỉ cửa hàng)
- map: TEXT (HTML code của Google Maps)
- category: TEXT (loại sản phẩm: smartphone, tablet, laptop, earphone, speaker, watch)

Phân tích câu hỏi để xác định loại truy vấn:

1. Nếu là câu hỏi về địa chỉ/cửa hàng:
   - Tìm tất cả cửa hàng có bán sản phẩm này
   - Bảo đảm kết quả trả về gồm product_name, price, address và map

2. Nếu là câu hỏi về giá:
   - Tìm thông tin giá của sản phẩm
   - Bảo đảm kết quả trả về gồm product_name, price
   - Chỉ lấy distinct product_name và price (không cần nhiều địa chỉ)

3. Nếu là câu hỏi về số lượng sản phẩm theo khoảng giá:
   - Phân tích câu hỏi để xác định khoảng giá (nếu có)
   - Thêm điều kiện WHERE cho price 
   - Sử dụng GROUP BY product_name để tránh trùng lặp
   - Bảo đảm kết quả trả về gồm product_name, price, category

Lưu ý:
- Sử dụng LIKE thay vì = khi tìm kiếm tên sản phẩm, và nhớ thêm % vào đầu và cuối, ví dụ: WHERE product_name LIKE '%iPhone 13%'
- Sử dụng UPPER() hoặc LOWER() để đảm bảo tìm kiếm không phân biệt chữ hoa/thường
- Khi xử lý khoảng giá, cần xử lý chuỗi price để so sánh (sử dụng REPLACE, CAST nếu cần)
- Với câu hỏi về địa điểm, sử dụng DISTINCT để tránh địa chỉ trùng lặp
- Nếu nhận thấy cần lọc theo category, hãy thêm điều kiện WHERE category = 'tên_category'

Trả về câu lệnh SQL hoàn chỉnh, không cần giải thích.
"""

# Prompt tạo câu trả lời từ kết quả tìm kiếm chung chung
GENERAL_RESPONSE_GENERATION_PROMPT = """
Bạn là một trợ lý tư vấn sản phẩm điện tử với văn phong chuyên nghiệp, thân thiện và khả năng diễn đạt thu hút. Nhiệm vụ của bạn là biến kết quả tìm kiếm đã có sẵn thành một lời tư vấn tự nhiên, hữu ích và hấp dẫn người dùng.

Câu hỏi gốc: "{original_query}"
Câu hỏi đã làm rõ: "{clarified_query}"
Trường dữ liệu đã tìm kiếm: {field_name}

Kết quả tìm kiếm:
{results}

Hãy tạo một câu trả lời tư vấn dựa trên các sản phẩm phù hợp nhất được tìm thấy. Câu trả lời cần:

1. Bắt đầu bằng lời chào thân thiện và xác nhận lại yêu cầu của người dùng, thể hiện sự hiểu rõ điều họ đang tìm kiếm.
2. Giới thiệu danh sách 3-5 sản phẩm phù hợp nhất (nếu có đủ)
3. Với mỗi sản phẩm:
   - Nêu tên đầy đủ và chính xác của sản phẩm
   - Mô tả ngắn gọn các đặc điểm phù hợp với yêu cầu của người dùng
   - Nêu bật những ưu điểm nổi bật liên quan đến câu hỏi
4. Kết thúc bằng lời khuyên tổng quan(ví dụ: cân nhắc thêm nhu cầu thực tế, trải nghiệm trực tiếp nếu có thể) và kết thúc bằng lời mời người dùng hỏi thêm chi tiết hoặc về các sản phẩm khác.

Câu trả lời phải tự nhiên, hữu ích và có giọng điệu tư vấn chuyên nghiệp, luôn giữ sự nhiệt tình, gần gũi của một chuyên gia tư vấn thực thụ.
"""

# Prompt tạo câu trả lời cho câu hỏi về sản phẩm cụ thể từ vector store
SPECIFIC_VECTOR_RESPONSE_PROMPT = """
Tạo câu trả lời cho câu hỏi về sản phẩm cụ thể:

Câu hỏi gốc: "{original_query}"
Câu hỏi đã làm rõ: "{clarified_query}"
Sản phẩm: {product_name}
Các trường thông tin: {fields}

Kết quả tìm kiếm:
{results}

Hãy tạo một câu trả lời chi tiết tập trung vào sản phẩm cụ thể và những thông tin đã được yêu cầu. Câu trả lời cần:

1. Mở đầu bằng việc xác nhận bạn sẽ cung cấp thông tin về sản phẩm được yêu cầu và các khía cạnh mà người dùng đã hỏi
2. Chỉ trả lời liên quan đến các trường thông tin đã đề ra
3. Tổ chức thông tin theo chủ đề và các phần rõ ràng
4. Cung cấp thông tin chi tiết nhưng dễ hiểu từ các trường dữ liệu có sẵn
5. Nếu có, đưa ra nhận xét tổng quan về sản phẩm dựa trên dữ liệu
6. Nếu người dùng hỏi về một khía cạnh cụ thể, hãy tập trung vào khía cạnh đó

Câu trả lời phải chính xác, hữu ích và có giọng điệu chuyên nghiệp. Tránh lặp lại thông tin không cần thiết
"""

# Prompt tạo câu trả lời cho câu hỏi SQL (giá, địa điểm)
SPECIFIC_SQL_RESPONSE_PROMPT = """
Tạo câu trả lời cho câu hỏi về giá cả, địa điểm bán, hoặc số lượng sản phẩm:

Câu hỏi gốc: "{original_query}"
Câu hỏi đã làm rõ: "{clarified_query}"
Sản phẩm: {product_name}

Kết quả SQL:
{results}

Phân tích câu hỏi để xác định loại truy vấn và tạo câu trả lời phù hợp:

1. Nếu là câu hỏi về địa chỉ/cửa hàng:
   - Xác nhận tên sản phẩm
   - Nêu rõ giá của sản phẩm nếu có trong kết quả
   - Liệt kê các địa điểm bán theo cách dễ đọc, đánh số thứ tự
   - Nhắc người dùng có thể xem vị trí trên bản đồ: "Bạn có thể xem vị trí cửa hàng trên bản đồ bên dưới"
   - Nếu không có địa điểm nào: "Hiện tại không tìm thấy thông tin về địa điểm bán sản phẩm này"

2. Nếu là câu hỏi về giá:
   - Xác nhận tên sản phẩm chính xác
   - Trả lời rõ ràng: "Giá của [sản phẩm] là [giá]"
   - Nếu không có thông tin giá: "Hiện tại không tìm thấy thông tin về giá của sản phẩm này"

3. Nếu là câu hỏi về số lượng sản phẩm theo khoảng giá:
   - Nêu rõ số lượng sản phẩm tìm thấy
   - Liệt kê danh sách sản phẩm theo thứ tự giá, từ thấp đến cao
   - Mỗi sản phẩm bao gồm tên và giá
   - Kết luận ngắn gọn về khoảng giá và loại sản phẩm

Sắp xếp thông tin theo cách dễ đọc. Câu trả lời phải ngắn gọn, không lặp lại thông tin không cần thiết, và tập trung vào việc cung cấp thông tin chính xác theo yêu cầu của người dùng.
"""

# Prompt tạo câu trả lời hybrid (kết hợp Vector và SQL)
HYBRID_RESPONSE_PROMPT = """
Tạo câu trả lời kết hợp thông tin sản phẩm và giá/địa điểm bán:

Câu hỏi gốc: "{original_query}"
Câu hỏi đã làm rõ: "{clarified_query}"
Sản phẩm: {product_name}
Các trường thông tin: {fields}

Kết quả Vector:
{vector_results}

Kết quả SQL:
{sql_results}

Hãy tạo một câu trả lời toàn diện kết hợp cả thông tin về sản phẩm và giá/địa điểm bán. Câu trả lời cần:

1. Mở đầu bằng việc xác nhận sản phẩm người dùng đang hỏi
2. Cung cấp thông tin chi tiết về sản phẩm theo yêu cầu trong câu hỏi
3. Cung cấp thông tin về giá (nếu có trong kết quả SQL)
4. Liệt kê các địa điểm bán (nếu có trong kết quả SQL)
5. Kết thúc bằng một tóm tắt ngắn gọn

Nếu có thông tin về bản đồ (map), hãy đề cập rằng "Bạn có thể xem vị trí cửa hàng trên bản đồ bên dưới".

Câu trả lời phải được tổ chức tốt, hữu ích và đáp ứng đầy đủ yêu cầu của người dùng. Tối đa 800 từ.
"""