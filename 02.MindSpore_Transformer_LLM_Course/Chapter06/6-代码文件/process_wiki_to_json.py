import json
import os

def parse_raw_file_to_json(file_path, num_items=None):
    """
    从外部文本文件中读取原始内容（支持.txt格式）
    参数：
        file_path：原始文本文件的路径（如"valkyria_raw.txt"）
    返回：读取成功返回原始文本字符串；失败则抛出异常并提示
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误：未找到原始文本文件，路径：{file_path}")

    # 读取文件内容（使用utf-8编码，适配多语言字符）
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text_lines = f.readlines()
            json_data = []
            text = ''
            idx = 0
            for text_line in text_lines:
                if text_line == ' \n' or text_line.startswith(' = '):
                    if text:
                        json_data.append({"id": idx, "text": text})
                        text = ''
                        idx += 1
                        if num_items and idx >= num_items:
                            return json_data
                    continue
                text += text_line

        # 简单校验内容是否为空
        if not json_data:
            raise ValueError("错误：没有获取到有效文本")
        return json_data
    except Exception as e:
        raise Exception(f"读取文件失败：{str(e)}")

def save_json_to_file(json_data, output_file="wiki.train.json"):
    """
    将解析后的JSON数据保存为文件（每行一个JSON对象，符合要求格式）
    参数：
        json_data：解析后的JSON数据列表
        output_file：输出JSON文件的路径（默认"valkyria_parsed.json"）
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for item in json_data:
                # 确保中文/特殊字符不转义，按行写入（便于后续读取和查看）
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")  # 每行一个JSON对象，避免压缩成一行
        print(f"✅ JSON文件保存成功！路径：{os.path.abspath(output_file)}")
    except Exception as e:
        raise Exception(f"保存JSON文件失败：{str(e)}")

if __name__ == "__main__":
    # --------------------------
    # 核心配置（用户需根据实际情况修改）
    RAW_TEXT_FILE_PATH = "wiki.train.tokens"  # 原始文本文件路径（如放在代码同目录）
    TARGET_NUM_ITEMS = 1000  # 控制输出数据条数（正整数，None则输出全部）
    SAVE_JSON_FILE_PATH = "wiki.train.json"
    # --------------------------
    
    print("=== 开始处理：从文件读取原始文本并转换为JSON ===")
    try:
        # 1. 从外部文件读取原始文本
        print(f"1/2 正在读取原始文本文件：{RAW_TEXT_FILE_PATH}")
        json_data = parse_raw_file_to_json(RAW_TEXT_FILE_PATH, num_items=TARGET_NUM_ITEMS)

        # 2. 保存JSON文件
        print(f"2/2 正在保存JSON文件（共{len(json_data)}条数据）")
        save_json_to_file(json_data, SAVE_JSON_FILE_PATH)

        # 输出处理结果摘要
        print("\n=== 处理完成！结果摘要 ===")
        print(f"📊 原始文本拆分后总段落数：{len([p for p in json_data])}")
        print(f"📄 输出JSON文件路径：{os.path.abspath(SAVE_JSON_FILE_PATH)}")
        print(f"🔍 第一条数据示例：\n{json.dumps(json_data[0], ensure_ascii=False, indent=2)}")

    except Exception as e:
        print(f"\n❌ 处理失败：{str(e)}")
