#!/usr/bin/env python3
"""
Batch Analysis Runner for Call Coverage Analysis
Processes multiple CSV files using Cantonese and Mandarin analyzers.
"""

import os
import pandas as pd
import json
import time
from datetime import datetime
import traceback

# Import analyzers as modules
try:
    import improved_call_coverage_checker as cantonese_analyzer
    print("✅ Cantonese analyzer loaded successfully")
except Exception as e:
    print(f"❌ Error loading Cantonese analyzer: {e}")
    cantonese_analyzer = None

try:
    import sys
    sys.path.append('Mandarin')
    import improved_call_coverage_checker_M as mandarin_analyzer
    print("✅ Mandarin analyzer loaded successfully")
except Exception as e:
    print(f"❌ Error loading Mandarin analyzer: {e}")
    mandarin_analyzer = None

# Import configuration
from config import (
    CONVERTED_TEXT_FOLDER,
    FILE_MAPPING_PATH,
    SCRIPT_FILE_PATH,
    OUTPUT_FOLDER,
    LOG_FILE_PATH,
    INCLUDE_SYSTEM_AUDIO_IN_ANALYSIS
)


def prepare_file_queue():
    """
    Prepare the queue of files to process by scanning the converted text folder
    and matching with the file mapping.
    
    Returns:
        list: List of task dictionaries with file_path, language, product_name, script_sheet
    """
    print("📋 Preparing file queue...")
    
    # Check if folders exist
    if not os.path.exists(CONVERTED_TEXT_FOLDER):
        print(f"❌ Error: Converted text folder not found: {CONVERTED_TEXT_FOLDER}")
        return []
    
    if not os.path.exists(FILE_MAPPING_PATH):
        print(f"❌ Error: File mapping not found: {FILE_MAPPING_PATH}")
        return []
    
    # Scan converted text folder for .csv files
    csv_files = []
    for filename in os.listdir(CONVERTED_TEXT_FOLDER):
        if filename.endswith('.csv'):
            csv_files.append(filename)
    
    print(f"📁 Found {len(csv_files)} CSV files in {CONVERTED_TEXT_FOLDER}")
    
    # Load file mapping
    try:
        mapping_df = pd.read_excel(FILE_MAPPING_PATH)
        print(f"📊 Loaded file mapping with {len(mapping_df)} entries")
    except Exception as e:
        print(f"❌ Error loading file mapping: {e}")
        return []
    
    # Process each CSV file
    task_queue = []
    for filename in csv_files:
        try:
            # 使用通用函数解析文件名
            task_info = parse_filename_for_task_info(filename, mapping_df)
            if not task_info:
                print(f"⚠️  Skipping file with parsing error: {filename}")
                continue
                
            # 直接使用解析结果
            sample_no = task_info['sample_no']
            language = task_info['language'] 
            product_name = task_info['product_name']
            call_type = task_info['call_type']
            script_sheet = task_info['script_sheet']
            
            print(f"📋 {call_type} file detected: {filename} -> Product: {product_name}, Type: {call_type}")
            
            # Create task
            task = {
                'file_path': os.path.join(CONVERTED_TEXT_FOLDER, filename),
                'language': language,
                'product_name': product_name,
                'script_sheet': script_sheet,
                'sample_no': sample_no,
                'filename': filename,
                'call_type': call_type  # Add call_type to task
            }
            
            task_queue.append(task)
            print(f"✅ Queued: {filename} -> Language: {language}, Product: {product_name}, Type: {call_type}, Sheet: {script_sheet}")
            
        except Exception as e:
            print(f"❌ Error processing file {filename}: {e}")
            continue
    
    print(f"🎯 Prepared {len(task_queue)} tasks for processing")
    return task_queue


def parse_filename_for_task_info(filename, mapping_df):
    """
    通用文件名解析函数 - 提取自main()函数中的现有逻辑
    
    Args:
        filename: 文件名 (可以是.csv或.xlsx)
        mapping_df: 产品映射DataFrame
        
    Returns:
        dict: 包含sample_no, language, product_name, call_type的字典，失败返回None
    """
    try:
        # 处理Excel文件名转换为CSV格式进行解析
        if filename.endswith('.xlsx'):
            filename = filename.replace('.xlsx', '.csv')
        
        # 复用现有逻辑：解析文件名
        base_name = filename.replace('.csv', '')
        if '.wav' in base_name:
            base_name = base_name.replace('.wav', '')
        
        # Split by underscore to get parts
        parts = base_name.split('_')
        if len(parts) < 3:
            return None
            
        # Extract sample number (first part before underscore)
        sample_no = parts[0]
        
        # Extract language from the last part
        language_part = parts[-1].upper()
        if language_part == 'C':
            language = 'CAN'
        elif language_part == 'M':
            language = 'MAN'
        else:
            return None
        
        # Check if this is a SQCCB file first (case-insensitive check)
        filename_upper = filename.upper()
        if 'SQCCB' in filename_upper:
            # For SQCCB files, set product_name directly without mapping lookup
            product_name = 'SQCCB'
            call_type = 'SQCCB'
        else:
            # For non-SQCCB files, look up product name in mapping
            call_type = 'Sales Call'
            sample_no_matches = mapping_df[mapping_df['Sample No'].astype(str) == str(sample_no)]
            if not sample_no_matches.empty:
                product_name = sample_no_matches.iloc[0]['Product Name']
            else:
                return None
        
        return {
            'sample_no': sample_no,
            'language': language,
            'product_name': product_name,
            'call_type': call_type,
            'script_sheet': f"{product_name}_{language}"
        }
        
    except Exception as e:
        return None


def seconds_to_mmss_format(seconds):
    """
    Convert seconds to MM:SS format for display.
    
    Args:
        seconds (int/float): Duration in seconds
        
    Returns:
        str: Formatted time string in MM:SS format
    """
    if pd.isna(seconds) or seconds == 0:
        return "00:00"
    
    try:
        total_seconds = int(float(seconds))
        minutes = total_seconds // 60
        remaining_seconds = total_seconds % 60
        return f"{minutes:02d}:{remaining_seconds:02d}"
    except (ValueError, TypeError):
        return "00:00"



def flag_outliers_iqr(df, column_name):
    """
    Flag outliers using IQR (Interquartile Range) method.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        column_name (str): Name of the column to analyze
        
    Returns:
        pd.Series: Series containing outlier flags
    """
    # Calculate Q1, Q3, and IQR
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Create flags
    flags = []
    for value in df[column_name]:
        if pd.isna(value):
            flags.append("")
        elif value < lower_bound:
            flags.append("Unusually Short")
        elif value > upper_bound:
            flags.append("Unusually Long")
        else:
            flags.append("")
    
    return pd.Series(flags)


def parse_time_to_seconds(time_str):
    """
    解析时间字符串为秒数
    支持格式: "00:18:30", "18:30", "1110.5", "1110"
    """
    if pd.isna(time_str):
        return 0.0
    
    time_str = str(time_str).strip()
    
    try:
        # 如果是纯数字，直接返回
        if time_str.replace('.', '').isdigit():
            return float(time_str)
        
        # 如果包含冒号，按时间格式解析
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(float, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:  # MM:SS
                minutes, seconds = map(float, parts)
                return minutes * 60 + seconds
        
        return 0.0
    except:
        return 0.0


def extract_data_from_multiple_sources_final(file_path, mapping_df):
    """
    修复版数据提取函数 - 直接从Executive Summary提取所有数据
    返回: (data_dict, error_message) 元组，成功时error_message为None
    """
    filename = os.path.basename(file_path)
    filename_base = os.path.splitext(filename)[0]
    
    # ========================================
    # 1. 修复文件名解析 - 转换Excel文件名为CSV格式
    # ========================================
    
    csv_filename = filename_base + '.csv'
    task_info = parse_filename_for_task_info(csv_filename, mapping_df)
    if not task_info:
        return None, f"文件名解析失败：'{filename}' 格式不符合 'sample_id_language' 规范"
        
    data = {
        'Sample No': task_info['sample_no'],
        'Language': task_info['language'],
        'Call Type': task_info['call_type']
    }
    
    # ========================================
    # 2. 修复Product Name - 使用mapping查找
    # ========================================
    
    sample_no = task_info['sample_no']
    try:
        sample_no_matches = mapping_df[mapping_df['Sample No'].astype(str) == str(sample_no)]
        if not sample_no_matches.empty:
            data['Product Name'] = sample_no_matches.iloc[0]['Product Name']
        else:
            return None, f"Sample No映射失败：Sample No '{sample_no}' 在mapping文件中不存在"
    except Exception as e:
        return None, f"Product Name查找出错：{str(e)}"
    
    # ========================================
    # 3. 修复Executive Summary解析 - 正确的标签匹配
    # ========================================
    
    try:
        xl = pd.ExcelFile(file_path)
        if 'Executive Summary' not in xl.sheet_names:
            return None, f"Excel文件缺少 'Executive Summary' sheet，可用sheets: {xl.sheet_names}"
        
        if 'Executive Summary' in xl.sheet_names:
            exec_summary = pd.read_excel(file_path, sheet_name='Executive Summary', header=None)
            
            # KPI区域数据提取
            for idx, row in exec_summary.iterrows():
                if pd.notna(row.iloc[0]):
                    cell_value = str(row.iloc[0]).strip()
                    
                    # 修正标签匹配 - 不带冒号的标签
                    if 'Compliance Coverage Rate (%)' in cell_value and pd.notna(row.iloc[1]):
                        coverage_str = str(row.iloc[1]).replace('%', '').strip()
                        try:
                            data['Coverage Rate (%)'] = float(coverage_str)
                        except:
                            data['Coverage Rate (%)'] = 0.0
                    
                    elif 'Total Points Checked' in cell_value and pd.notna(row.iloc[1]):
                        try:
                            data['Total Points'] = int(row.iloc[1])
                        except:
                            data['Total Points'] = 0
                    
                    elif 'Covered Points' in cell_value and pd.notna(row.iloc[1]):
                        try:
                            data['Covered Points'] = int(row.iloc[1])
                        except:
                            data['Covered Points'] = 0
                    
                    elif 'Total Call Duration' in cell_value and pd.notna(row.iloc[1]):
                        duration_str = str(row.iloc[1]).strip()
                        
                        # 处理时长格式
                        if ':' in duration_str:
                            # 转换为秒数
                            data['Total Duration (s)'] = parse_time_to_seconds(duration_str)
                        else:
                            # 纯秒数格式
                            try:
                                data['Total Duration (s)'] = float(duration_str.replace('s', ''))
                            except:
                                data['Total Duration (s)'] = 0.0
            
            # ========================================
            # 4. Speaker View数据提取
            # ========================================
            
            speaker_view_start = None
            for idx, row in exec_summary.iterrows():
                if pd.notna(row.iloc[0]) and 'Key Insights - Speaker View' in str(row.iloc[0]):
                    speaker_view_start = idx
                    break
            
            if speaker_view_start is not None:
                # 查找表头行
                header_row = None
                for idx in range(speaker_view_start + 1, min(speaker_view_start + 10, len(exec_summary))):
                    row = exec_summary.iloc[idx]
                    if pd.notna(row.iloc[0]) and 'Role' in str(row.iloc[0]):
                        header_row = idx
                        break
                
                if header_row is not None:
                    # 解析表头找到列位置
                    headers = exec_summary.iloc[header_row].fillna('').astype(str)
                    role_col = None
                    word_count_col = None
                    
                    for col_idx, header in enumerate(headers):
                        if 'Role' in header:
                            role_col = col_idx
                        elif 'Word Count' in header:
                            word_count_col = col_idx
                    
                    if role_col is not None and word_count_col is not None:
                        # 提取Sales和Customer的Word Count
                        for idx in range(header_row + 1, min(header_row + 10, len(exec_summary))):
                            row = exec_summary.iloc[idx]
                            if pd.notna(row.iloc[role_col]) and pd.notna(row.iloc[word_count_col]):
                                role = str(row.iloc[role_col]).strip()
                                word_count_str = str(row.iloc[word_count_col]).strip()
                                
                                try:
                                    word_count = int(word_count_str)
                                    if role == 'Sales':
                                        data['Sales Word Count'] = word_count
                                    elif role == 'Customer':
                                        data['Customer Word Count'] = word_count
                                except:
                                    continue
                        
    except Exception as e:
        return None, f"Excel文件读取失败：{str(e)}"
    
    # 设置默认值
    defaults = {
        'Coverage Rate (%)': 0.0, 'Total Points': 0, 'Covered Points': 0,
        'Total Duration (s)': 0.0, 'Sales Word Count': 0, 'Customer Word Count': 0
    }
    for key, default_value in defaults.items():
        if key not in data:
            data[key] = default_value
    
    return data, None


def generate_summarized_report_final():
    """
    最终版汇总报告生成函数
    """
    print("🚀 Generating Summarized Report from All Historical Data")
    
    # 创建日志session
    import datetime
    log_session_start = f"\n{'='*80}\n📊 SUMMARIZED REPORT SESSION - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n"
    
    # 写入日志文件
    try:
        with open('batch_analysis.log', 'a', encoding='utf-8') as log_file:
            log_file.write(log_session_start)
    except:
        pass  # 如果日志写入失败，不影响主要功能
    
    # 读取mapping文件（复用现有逻辑）
    try:
        mapping_df = pd.read_excel(FILE_MAPPING_PATH)
        log_msg = f"✅ Successfully loaded mapping file: {FILE_MAPPING_PATH} ({len(mapping_df)} records)\n"
        print(f"✅ Loaded mapping file with {len(mapping_df)} records")
    except Exception as e:
        error_msg = f"❌ Error reading mapping file: {e}"
        log_msg = f"❌ Error reading mapping file {FILE_MAPPING_PATH}: {str(e)}\n"
        print(error_msg)
        try:
            with open('batch_analysis.log', 'a', encoding='utf-8') as log_file:
                log_file.write(log_msg)
        except:
            pass
        return
    
    # 扫描所有Excel输出文件
    excel_files = [f for f in os.listdir(OUTPUT_FOLDER) 
                   if f.endswith('.xlsx') and not f.startswith('Summarized_')]
    
    if not excel_files:
        error_msg = "❌ No existing output files found."
        log_msg = f"❌ No Excel output files found in {OUTPUT_FOLDER}\n"
        print(error_msg)
        try:
            with open('batch_analysis.log', 'a', encoding='utf-8') as log_file:
                log_file.write(log_msg + log_msg)
        except:
            pass
        return
    
    log_msg += f"📁 Found {len(excel_files)} Excel files to process\n\n"
    
    all_results = []
    success_count = 0
    error_count = 0
    
    for excel_file in excel_files:
        try:
            file_path = os.path.join(OUTPUT_FOLDER, excel_file)
            
            # 使用改进的数据提取函数
            data, error_message = extract_data_from_multiple_sources_final(file_path, mapping_df)
            if data is not None:
                all_results.append(data)
                success_count += 1
                success_msg = f"✅ Extracted data from: {excel_file}"
                log_msg += f"✅ SUCCESS: {excel_file}\n"
                print(success_msg)
            else:
                error_count += 1
                error_msg = f"⚠️  Could not extract data from: {excel_file}"
                detailed_error = f"   原因: {error_message}" if error_message else ""
                log_msg += f"❌ FAILED: {excel_file}\n   原因: {error_message or 'Unknown error'}\n"
                print(error_msg)
                if detailed_error:
                    print(detailed_error)
                
        except Exception as e:
            error_count += 1
            error_msg = f"⚠️  Error processing {excel_file}: {e}"
            log_msg += f"❌ EXCEPTION: {excel_file}\n   错误: {str(e)}\n"
            print(error_msg)
    
    # 写入处理结果到日志
    log_msg += f"\n📊 处理结果统计:\n"
    log_msg += f"   成功: {success_count} 个文件\n"
    log_msg += f"   失败: {error_count} 个文件\n"
    log_msg += f"   总计: {len(excel_files)} 个文件\n\n"
    
    try:
        with open('batch_analysis.log', 'a', encoding='utf-8') as log_file:
            log_file.write(log_msg)
    except:
        pass
    
    if not all_results:
        final_error = "❌ No valid data extracted from any files."
        print(final_error)
        try:
            with open('batch_analysis.log', 'a', encoding='utf-8') as log_file:
                log_file.write(f"{final_error}\n")
        except:
            pass
        return
    
    # 转换为DataFrame
    consolidated_df = pd.DataFrame(all_results)
    
    # 添加格式化时长列（复用现有函数）
    consolidated_df['Total Duration (MM:SS)'] = consolidated_df['Total Duration (s)'].apply(seconds_to_mmss_format)
    
    # ========================================
    # 关键：在完整历史数据集上进行异常值检测
    # ========================================
    
    print("🔍 Applying outlier detection using IQR method on complete historical dataset...")
    consolidated_df['Duration Outlier'] = flag_outliers_iqr(consolidated_df, 'Total Duration (s)')
    consolidated_df['Sales WC Outlier'] = flag_outliers_iqr(consolidated_df, 'Sales Word Count')
    consolidated_df['Customer WC Outlier'] = flag_outliers_iqr(consolidated_df, 'Customer Word Count')
    
    # 复用现有的列排序
    column_order = [
        'Sample No', 'Product Name', 'Language', 'Call Type',
        'Coverage Rate (%)', 'Total Points', 'Covered Points',
        'Total Duration (s)', 'Total Duration (MM:SS)', 'Duration Outlier',
        'Sales Word Count', 'Sales WC Outlier',
        'Customer Word Count', 'Customer WC Outlier'
    ]
    consolidated_df = consolidated_df[column_order]
    
    # 保存报告
    summary_report_path = os.path.join(OUTPUT_FOLDER, "Summarized_Analysis_Report.xlsx")
    consolidated_df.to_excel(summary_report_path, index=False, sheet_name='Summary Report')
    
    # 复用现有的统计输出逻辑
    duration_outliers = len(consolidated_df[consolidated_df['Duration Outlier'] != ''])
    sales_outliers = len(consolidated_df[consolidated_df['Sales WC Outlier'] != ''])
    customer_outliers = len(consolidated_df[consolidated_df['Customer WC Outlier'] != ''])
    
    print(f"✅ Summarized report generated successfully!")
    print(f"   📁 File: {summary_report_path}")
    print(f"   📊 Contains {len(consolidated_df)} historical analysis results")
    print(f"   🎯 Outliers detected:")
    print(f"      • Duration outliers: {duration_outliers}")
    print(f"      • Sales word count outliers: {sales_outliers}")
    print(f"      • Customer word count outliers: {customer_outliers}")
    
    # 写入成功信息到日志
    final_log = f"✅ 汇总报告生成成功!\n"
    final_log += f"   文件路径: {summary_report_path}\n"
    final_log += f"   包含记录: {len(consolidated_df)} 条\n"
    final_log += f"   异常值检测: Duration({duration_outliers}), Sales WC({sales_outliers}), Customer WC({customer_outliers})\n"
    final_log += f"{'='*80}\n"
    
    try:
        with open('batch_analysis.log', 'a', encoding='utf-8') as log_file:
            log_file.write(final_log)
    except:
        pass
    print(f"   🎯 Outliers detected:")
    print(f"      • Duration outliers: {duration_outliers}")
    print(f"      • Sales word count outliers: {sales_outliers}")
    print(f"      • Customer word count outliers: {customer_outliers}")


def main():
    """
    Main processing loop for batch analysis.
    """
    print("🚀 Starting Batch Call Coverage Analysis")
    print("=" * 50)
    
    start_time = datetime.now()
    logs = []
    summarized_results = []  # Initialize empty list for summary data collection
    
    # Prepare file queue
    task_queue = prepare_file_queue()
    if not task_queue:
        print("❌ No files to process. Exiting.")
        return
    
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Process each task
    successful_tasks = 0
    failed_tasks = 0
    
    for i, task in enumerate(task_queue, 1):
        print(f"\n📋 Processing task {i}/{len(task_queue)}: {task['filename']}")
        print("-" * 40)
        
        task_start_time = datetime.now()
        
        try:
            # Route to appropriate analyzer based on language
            if task['language'] == 'CAN':
                if cantonese_analyzer is None:
                    raise ValueError("Cantonese analyzer not available")
                print("🔤 Using Cantonese analyzer...")
                result = cantonese_analyzer.run_analysis(
                    call_file_path=task['file_path'],
                    script_file_path=SCRIPT_FILE_PATH,
                    script_sheet_name=task['script_sheet'],
                    output_folder=OUTPUT_FOLDER,
                    call_type=task['call_type'],
                    language=task['language'],
                    include_system_audio=INCLUDE_SYSTEM_AUDIO_IN_ANALYSIS
                )
            elif task['language'] == 'MAN':
                if mandarin_analyzer is None:
                    raise ValueError("Mandarin analyzer not available")
                print("🔤 Using Mandarin analyzer...")
                result = mandarin_analyzer.run_analysis(
                    call_file_path=task['file_path'],
                    script_file_path=SCRIPT_FILE_PATH,
                    script_sheet_name=task['script_sheet'],
                    output_folder=OUTPUT_FOLDER,
                    call_type=task['call_type'],
                    language=task['language'],
                    include_system_audio=INCLUDE_SYSTEM_AUDIO_IN_ANALYSIS
                )
            else:
                raise ValueError(f"Unsupported language: {task['language']}")
            
            # Check result status
            if result['status'] == 'SUCCESS':
                successful_tasks += 1
                task_duration = (datetime.now() - task_start_time).total_seconds()
                
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'filename': task['filename'],
                    'sample_no': task['sample_no'],
                    'language': task['language'],
                    'product_name': task['product_name'],
                    'script_sheet': task['script_sheet'],
                    'status': 'SUCCESS',
                    'output_file': result['output_file'],
                    'coverage_rate': result['coverage_rate'],
                    'processing_time': task_duration,
                    'total_points': result.get('total_points', 0),
                    'covered_points': result.get('covered_points', 0)
                }
                
                # Create summary row for the consolidated report
                # Process Language column: Convert 'CAN', 'MAN', 'ENG' to full words
                language_full = {
                    'CAN': 'Cantonese',
                    'MAN': 'Mandarin',
                    'ENG': 'English'
                }.get(task['language'], task['language'])
                
                summary_row = {
                    'Sample No': task['sample_no'],
                    'Product Name': task['product_name'],
                    'Language': language_full,
                    'Call Type': task['call_type'],
                    'Coverage Rate (%)': result['coverage_rate'],
                    'Total Points': result.get('total_points', 0),
                    'Covered Points': result.get('covered_points', 0),
                    'Total Duration (s)': result.get('total_call_duration', 0),
                    'Sales Word Count': result.get('sales_word_count', 0),
                    'Customer Word Count': result.get('customer_word_count', 0)
                }
                
                # Append to summarized_results list using edge-run-edge strategy
                summarized_results.append(summary_row)
                
                print(f"✅ Task completed successfully!")
                print(f"   Output: {result['output_file']}")
                print(f"   Coverage: {result['coverage_rate']:.1f}%")
                print(f"   Duration: {task_duration:.1f}s")
                print(f"   Added to summary report: Sample {task['sample_no']}, Language: {language_full}")
                
            else:
                failed_tasks += 1
                task_duration = (datetime.now() - task_start_time).total_seconds()
                
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'filename': task['filename'],
                    'sample_no': task['sample_no'],
                    'language': task['language'],
                    'product_name': task['product_name'],
                    'script_sheet': task['script_sheet'],
                    'status': 'FAILED',
                    'error': result.get('error', 'Unknown error'),
                    'processing_time': task_duration
                }
                
                print(f"❌ Task failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            failed_tasks += 1
            task_duration = (datetime.now() - task_start_time).total_seconds()
            error_details = str(e)
            
            # Get traceback for debugging
            tb_str = traceback.format_exc()
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'filename': task['filename'],
                'sample_no': task['sample_no'],
                'language': task['language'],
                'product_name': task['product_name'],
                'script_sheet': task['script_sheet'],
                'status': 'ERROR',
                'error': error_details,
                'traceback': tb_str,
                'processing_time': task_duration
            }
            
            print(f"💥 Exception occurred: {error_details}")
            print(f"   See log file for full traceback")
        
        # Add log entry
        logs.append(log_entry)
    
    # Generate Summarized Report (after all individual tasks are complete)
    print("\n" + "=" * 50)
    print("📊 Generating Summarized Analysis Report...")
    
    # 使用改进的汇总报告生成函数
    generate_summarized_report_final()
    
    # Write logs to file
    total_duration = (datetime.now() - start_time).total_seconds()
    
    summary_log = {
        'batch_summary': {
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration_seconds': total_duration,
            'total_tasks': len(task_queue),
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': (successful_tasks / len(task_queue) * 100) if task_queue else 0
        },
        'task_logs': logs
    }
    
    try:
        with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(summary_log, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Log saved to: {LOG_FILE_PATH}")
    except Exception as e:
        print(f"❌ Error saving log file: {e}")
    
    # Print final summary
    print("\n" + "=" * 50)
    print("🎉 Batch Analysis Completed!")
    print(f"📊 Summary:")
    print(f"   • Total tasks: {len(task_queue)}")
    print(f"   • Successful: {successful_tasks}")
    print(f"   • Failed: {failed_tasks}")
    print(f"   • Success rate: {(successful_tasks / len(task_queue) * 100):.1f}%")
    print(f"   • Total duration: {total_duration:.1f}s")
    print(f"   • Output folder: {OUTPUT_FOLDER}")
    print(f"   • Log file: {LOG_FILE_PATH}")
    
    # Check if summarized report was generated
    summary_report_path = os.path.join(OUTPUT_FOLDER, "Summarized_Analysis_Report.xlsx")
    if os.path.exists(summary_report_path):
        print(f"   • Summarized report: {summary_report_path}")
    
    print("\n🔍 Files Generated:")
    print(f"   📋 Individual Analysis: {successful_tasks} Excel files with Executive Summary, Call Text Analysis, and Sentence Level Analysis")
    if os.path.exists(summary_report_path):
        print(f"   📊 Consolidated Report: Summarized_Analysis_Report.xlsx with cross-call summary data")
    print(f"   📄 Batch Log: {os.path.basename(LOG_FILE_PATH)} with detailed processing logs")


def generate_summarized_report_only():
    """
    Generate summarized report from all existing output files without running new analysis.
    Useful when you want to update the summarized report after running separate batches.
    """
    # 直接调用改进的汇总报告生成函数
    generate_summarized_report_final()


if __name__ == "__main__":
    import sys
    
    # Check if user wants to generate summarized report only
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize-only":
        generate_summarized_report_only()
    else:
        main()
