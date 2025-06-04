TONES = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS = ["aˇ", "aˆ", "Aˇ", "Aˆ", "eˆ", "Eˆ", "oˆ", "o˒", "Oˆ", "O˒", "u˒", "U˒", "D-", "d‑"]

dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỮiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"

def make_groups():
    groups = []
    i = 0
    while i < len(dictionary) - 5:
        group = [c for c in dictionary[i:i + 6]]
        i += 6
        groups.append(group)
    return groups

groups = make_groups()

def correct_tone_position(word):
    word = word[:-1]  # Bỏ ký hiệu dấu
    if len(word) < 2:
        return word[-1] if word else ""
    first_ord_char = ""
    second_order_char = ""
    for char in word:
        for group in groups:
            if char in group:
                second_order_char = first_ord_char
                first_ord_char = group[0]
    if word[-1] == first_ord_char and second_order_char != "":
        pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
        for pair in pair_chars:
            if pair in word and second_order_char in ["u", "U", "i", "I"]:
                return first_ord_char
        return second_order_char
    return first_ord_char

def decoder(recognition):
    for char in TARGETS:
        recognition = recognition.replace(char, SOURCES[TARGETS.index(char)])
    if len(recognition) < 1:
        return recognition
    if recognition[-1] in TONES:
        if len(recognition) < 2:
            return recognition
        replace_char = correct_tone_position(recognition)
        tone = recognition[-1]
        recognition = recognition[:-1]
        for group in groups:
            if replace_char in group:
                recognition = recognition.replace(replace_char, group[TONES.index(tone)])
    return recognition.lower()  # Chuyển về chữ thường


# Đọc file từ điển và giải mã
input_file = "./vn_dictionaryuu.txt"
output_file = "./vn_dictionary.txt"

with open(input_file, 'r', encoding='utf-8') as f_in:
    words = [line.strip() for line in f_in if line.strip()]  # Bỏ các dòng trống

decoded_words = [decoder(word) for word in words]

# Ghi kết quả vào file mới
with open(output_file, 'w', encoding='utf-8') as f_out:
    for decoded in decoded_words:
        f_out.write(decoded + '\n')

print(f"Đã giải mã và chuyển về chữ thường, lưu kết quả vào: {output_file}")