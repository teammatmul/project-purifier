import re
import emoji

def preprocess_string(text):
    """
    입력받은 text 를 전처리 하는 함수.

    :param text: str
    :return : str

    """

    # 이모티콘부터 제거
    no_emoticon = ''
    for char in text:
        if char not in emoji.UNICODE_EMOJI:
            no_emoticon += char

    # 특수문자 기준 split
    no_punctuation = re.split(r'([!,?]+)|([.]+)|([,]+)|(["])|([\'])|([&]+)|([(]+)|([)]+)|([~]+)|([♡]+)|([☆,★]+)',
                              no_emoticon.strip())
    no_punctuation_text = []

    for string in no_punctuation:
        if (string == '') or (string is None): continue
        no_punctuation_text.append(string)

    no_punctuation_text = ' '.join(no_punctuation_text)

    # 단독으로 쓰인 자모음 분리
    split_char = re.split(r'([ㄱ-ㅣ0-9]+)', no_punctuation_text.strip())
    split_char = ' '.join(split_char)

    # 한국어에서 단독으로 자주 쓰이는 자모음 뭉치 분리
    split_char = re.split(r'([ㅎ]{2,})|([ㅜ,ㅠ]{2,})|([ㅗ]+)|([ㅋ,ㄱ,ㄲ]{2,})|\s+', split_char.strip())
    final_text = []
    for string in split_char:
        if (string == '') or (string is None): continue
        final_text.append(string)

    return ' '.join(final_text)