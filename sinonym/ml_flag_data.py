"""Linguistic character sets for Chinese vs Japanese name classification.

These constants define the heuristic-flag features of the trained classifier.
They live in this dependency-free module so the runtime fast scorer
(``sinonym.ml_fast_scorer``) can use them without importing sklearn/scipy,
while the trained transformer (``sinonym.ml_model_components``) re-exports
them for training and artifact deserialization.
"""

HEURISTIC_FLAG_NAMES = (
    "jp_iter_mark",
    "jp_surname_chars",
    "cn_surname_chars",
    "jp_name_endings",
    "cn_name_endings",
    "jp_unique_chars",
    "cn_simplified_chars",
    "len_eq2",
    "len_eq3",
    "len_ge4",
    "jp_frequent_chars",
    "cn_frequent_chars",
    "surname_jp_pattern",
    "surname_cn_pattern",
    "given_jp_pattern",
    "given_cn_pattern",
    "jp_ending_ratio",
    "cn_ending_ratio",
    "char_diversity",
    "avg_char_strokes",
)

# fmt: off
CN_SURNAME_CHARS = {
    "王", "李", "张", "刘", "陈", "杨", "黄", "赵", "周", "吴",
    "徐", "孙", "朱", "马", "胡", "郭", "林", "何", "高", "梁",
    "郑", "罗", "宋", "谢", "唐", "韩", "曹", "许", "邓", "萧",
}

JP_SURNAME_CHARS = {
    "田", "中", "山", "本", "木", "村", "井", "川", "原", "藤",
    "野", "池", "石", "松", "竹", "林", "森", "東", "西", "北",
    "南", "上", "下", "大", "小", "高", "長", "新", "古", "佐",
}

JP_NAME_ENDINGS = {
    "子", "美", "也", "郎", "男", "之", "哉", "奈", "菜", "里",
    "佳", "香", "恵", "愛", "花", "夏", "春", "秋", "冬", "雪",
    "月", "星", "海", "空", "光", "希", "未", "真", "純", "清",
}

CN_NAME_ENDINGS = {
    "华", "明", "伟", "强", "军", "平", "勇", "杰", "涛", "波",
    "磊", "鹏", "辉", "刚", "超", "飞", "龙", "凤", "霞", "红",
    "玲", "丽", "娟", "芳", "燕", "静", "敏", "慧", "兰", "梅",
}

ITERATION_MARK = "々"

JP_UNIQUE_CHARS = {
    "辺", "沢", "浜", "栄", "竜", "礼", "稲", "彦", "蔵", "衛",
    "介", "助", "郎", "丸", "丞", "斎", "斉", "桜", "櫻",
}

CN_SIMPLIFIED_CHARS = {
    "赵", "刘", "陈", "邓", "关", "兰", "乔", "许", "闫", "贾",
    "钱", "孔", "白", "崔", "康", "史", "顾", "侯", "邵", "孟",
}

CN_FREQUENT_CHARS = {
    "国", "民", "建", "文", "志", "忠", "义", "礼", "智", "信",
    "仁", "勇", "才", "德", "宝", "福", "寿", "康", "安", "宁",
}

JP_FREQUENT_CHARS = {
    "雄", "雅", "正", "直", "克", "修", "治", "和", "昭", "博",
    "弘", "宏", "広", "寛", "豊", "富", "貴", "尊", "敬", "慶",
}
# fmt: on
