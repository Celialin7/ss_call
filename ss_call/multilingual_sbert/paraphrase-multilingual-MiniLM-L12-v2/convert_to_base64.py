import base64
import os

def convert_to_base64(input_file, output_file):
    # 读取二进制文件
    with open(input_file, 'rb') as f:
        binary_data = f.read()
    
    # 转换为Base64编码
    base64_data = base64.b64encode(binary_data)
    
    # 将Base64编码保存为文本文件
    with open(output_file, 'w') as f:
        f.write(base64_data.decode('ascii'))

def main():
    # 获取当前目录下所有model_part_*文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_parts = [f for f in os.listdir(current_dir) if f.startswith('model_part_')]
    
    # 为每个文件创建对应的Base64文本文件
    for part in model_parts:
        input_file = os.path.join(current_dir, part)
        output_file = os.path.join(current_dir, f'{part}.txt')
        print(f'Converting {part} to Base64...')
        convert_to_base64(input_file, output_file)
        print(f'Created {output_file}')

if __name__ == '__main__':
    main()