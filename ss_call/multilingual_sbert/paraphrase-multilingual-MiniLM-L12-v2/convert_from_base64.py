import base64
import os

def convert_from_base64(input_file, output_file):
    # 读取Base64文本文件
    with open(input_file, 'r') as f:
        base64_data = f.read()
    
    # 转换回二进制数据
    binary_data = base64.b64decode(base64_data)
    
    # 写入二进制文件
    with open(output_file, 'wb') as f:
        f.write(binary_data)

def merge_binary_files(binary_files, output_file):
    # 合并所有二进制文件
    with open(output_file, 'wb') as outfile:
        for binary_file in binary_files:
            with open(binary_file, 'rb') as infile:
                outfile.write(infile.read())

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取所有.txt文件（Base64编码的文件）
    base64_files = [f for f in os.listdir(current_dir) if f.endswith('.txt') and f.startswith('model_part_')]
    base64_files.sort()  # 确保按正确顺序处理文件
    
    # 临时二进制文件列表
    temp_binary_files = []
    
    # 转换每个Base64文件回二进制
    for base64_file in base64_files:
        input_file = os.path.join(current_dir, base64_file)
        temp_output = os.path.join(current_dir, f'temp_{base64_file[:-4]}')  # 移除.txt后缀
        print(f'Converting {base64_file} back to binary...')
        convert_from_base64(input_file, temp_output)
        temp_binary_files.append(temp_output)
    
    # 合并所有临时二进制文件
    final_output = os.path.join(current_dir, 'pytorch_model.bin')
    print('Merging binary files...')
    merge_binary_files(temp_binary_files, final_output)
    
    # 清理临时文件
    for temp_file in temp_binary_files:
        os.remove(temp_file)
        print(f'Removed temporary file: {temp_file}')
    
    print(f'Successfully created {final_output}')

if __name__ == '__main__':
    main()