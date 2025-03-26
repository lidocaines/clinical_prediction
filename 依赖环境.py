def filter_requirements(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            lines = input_file.readlines()

        # 过滤掉本地编译的包
        pypi_lines = []
        for line in lines:
            if not line.strip().endswith(('.whl', '.egg')) and '@ file:///' not in line:
                pypi_lines.append(line)

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.writelines(pypi_lines)

        print(f"已过滤并保存到 {output_file_path}")
    except FileNotFoundError:
        print(f"未找到文件: {input_file_path}")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    input_file = 'requirements.txt'
    output_file = 'requirements_pypi.txt'
    filter_requirements(input_file, output_file)