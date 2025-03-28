# Yardımcı fonksiyonlar
def create_color_mask(overlay_path, output_path):
    # Overlay görüntüsünü oku
    overlay = cv2.imread(overlay_path)

    # Görüntüyü HSV renk uzayına çevir
    hsv = cv2.cvtColor(overlay, cv2.COLOR_BGR2HSV)

    # Kırmızı ve yeşil renkler için maske oluştur
    # Kırmızı renk için iki ayrı aralık (HSV renk uzayında)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Yeşil renk için aralık
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])

    # Kırmızı ve yeşil için maskeler
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Birleşik maske (kırmızı veya yeşil)
    mask = cv2.bitwise_or(mask_red, mask_green)

    # Maskeyi beyaz-siyah formatına çevir
    mask_binary = np.zeros_like(mask)
    mask_binary[mask > 0] = 255

    # Maskeyi kaydet
    cv2.imwrite(output_path, mask_binary)
    print(f"Maske oluşturuldu: {output_path}")

def process_overlay_folder(input_folder, output_folder):
    # Çıktı klasörünü oluştur
    os.makedirs(output_folder, exist_ok=True)

    # Giriş klasöründeki tüm dosyaları işle
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '_mask.png'
            output_path = os.path.join(output_folder, output_filename)

            create_color_mask(input_path, output_path)
