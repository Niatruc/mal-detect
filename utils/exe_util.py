import lief
import random
import struct
from functools import reduce

def find_pe_modifiable_range(exe_file_path, max_len = 2**20, use_range=0b1111):
    exe_info = lief.parse(exe_file_path)

    if not exe_info:
        print("该文件可能为打包的文件")
        modifiable_range_selection = find_packed_pe_modifiable_range(exe_file_path, max_len)
    else:
        dos_header_modifiable_range1 = (2, 0x40 - 4)

        pe_header_offset = exe_info.dos_header.addressof_new_exeheader
        dos_header_modifiable_range2 = (0x40, pe_header_offset)

        # TimeDateStamp(4), PointerToSymbolTable(4), NumberOfSymbols(4)
        pe_header_modifiable_range1 = (pe_header_offset + 8, pe_header_offset + 8 + 12)

        image_optional_header_offset = pe_header_offset + 24

        # MajorLinkerVersion(1), MinorLinkerVersion(1), SizeOfCode(4), SizeOfInitializedData(4), SizeOfUninitializedData(4)
        pe_header_modifiable_range2 = (image_optional_header_offset + 2, image_optional_header_offset + 2 + 14)

        # 64位PE文件的ImageBase字段占8个字节
        next1 = 24
        # if exe_info.header.is_64:
        if exe_info.optional_header.magic == lief.PE.PE_TYPE.PE32_PLUS:
            next1 = 28

        # MajorOperatingSystemVersion(2), MinorOperatingSystemVersion(2), MajorImageVersion(2), MinorImageVersion(2)
        pe_header_modifiable_range3 = (image_optional_header_offset + 2 + 14 + next1, image_optional_header_offset + 2 + 14 + next1 + 8)

        # Win32VersionValue(4)
        pe_header_modifiable_range4 = (image_optional_header_offset + 2 + 14 + next1 + 8 + 4, image_optional_header_offset + 2 + 14 + next1 + 8 + 4 + 4)

        checksum_offset = image_optional_header_offset + 2 + 14 + next1 + 8 + 4 + 4 + 8

        # CheckSum(4)
        pe_header_modifiable_range5 = (checksum_offset, checksum_offset + 4)

        # 64位PE文件的SizeOfStackReserve等4个字段各占8个字节
        next2 = 20
        # if exe_info.header.is_64:
        if exe_info.optional_header.magic == lief.PE.PE_TYPE.PE32_PLUS:
            next2 = 20 + 4 * 4

        # LoaderFlags(4)
        pe_header_modifiable_range6 = (checksum_offset + 4 + next2, checksum_offset + 4 + next2 + 4)

        image_optional_header_end_offset = image_optional_header_offset + exe_info.header.sizeof_optional_header

        # 因头部补齐而产生的剩余可改空间
        header_end_modifiable_range = (image_optional_header_end_offset + 40 * exe_info.header.numberof_sections, exe_info.sizeof_headers)

        # 各区块因补齐而产生的剩余可改空间
        pe_modifiable_sections_range_list = []
        for sec in exe_info.sections:
            if sec.size <= sec.virtual_size:
                continue
            if sec.offset + sec.virtual_size >= max_len:
                break
            pe_modifiable_sections_range_list.append((sec.offset + sec.virtual_size, min(sec.offset + sec.size, max_len)))

        modifiable_range_selection = [
            [
                dos_header_modifiable_range1,
                dos_header_modifiable_range2
            ],
            [
                pe_header_modifiable_range1,
                pe_header_modifiable_range2,
                pe_header_modifiable_range3,
                pe_header_modifiable_range4,
                pe_header_modifiable_range5,
                pe_header_modifiable_range6,
            ],
            [header_end_modifiable_range],
            pe_modifiable_sections_range_list
        ]

    # 根据use_range(形如0b0111)选出要用的范围
    final_selected_modifiable_range = []
    for i, part in enumerate(modifiable_range_selection):
        if use_range == 0:
            break

        select_part = use_range % 2
        use_range //= 2
        if select_part:
            final_selected_modifiable_range += modifiable_range_selection[i]

    final_selected_modifiable_range2 = []
    for bound in final_selected_modifiable_range:
        if bound[0] < bound[1]:
            final_selected_modifiable_range2.append(bound)

    return final_selected_modifiable_range2

# 对于被加密且需要其他程序解压的PE文件,其PE头部一般是全零(解压后才被替换), 整个PE头和DOS头(除魔数MZ外)都可改.
def find_packed_pe_modifiable_range(exe_file_path, max_len = 2**20, use_range=0b1111):
    # doc = open(exe_file_path, 'rb').read()
    # bytes_list = [byte for byte in doc]
    dos_header_modifiable_range = (2, 0x40)
    pe_header_modifiable_range = (0x40, 0x40 + 4 + 20 + 0xe0) # 使用32位PE头大小, 确保不会改动打包部分
    return [[dos_header_modifiable_range], [pe_header_modifiable_range]]

def find_pe_sec_modifiable_range(exe_file_path, max_len = 2**20, use_range=0b1111):
    exe_info = lief.parse(exe_file_path)

    # 各区块因补齐而产生的剩余可改空间
    pe_modifiable_sections_range_list = []
    for sec in exe_info.sections:
        if sec.size <= sec.virtual_size:
            continue
        if sec.offset + sec.virtual_size >= max_len:
            break
        pe_modifiable_sections_range_list.append((sec.offset + sec.virtual_size, min(sec.offset + sec.size, max_len)))

def get_modifiable_range_list(fn, changed_range, changed_bytes_cnt):
    modifiable_range_list = []
    # 从可改的第一个字节开始到第changed_bytes_cnt个字节结束
    modifiable_range_list = find_pe_modifiable_range(fn, use_range=changed_range)
    if changed_bytes_cnt > 0:
        cbc = changed_bytes_cnt
        mrl = []
        for bound in modifiable_range_list:
            bound_len = bound[1] - bound[0]
            if cbc < bound_len:
                mrl.append((bound[0], bound[0] + cbc))
                cbc = 0
                break
            else:
                mrl.append(bound)
                cbc -= bound_len
        modifiable_range_list = mrl

    return modifiable_range_list

# 将val值转为小端字节数组. byte_cnt指定赋值的字段有多少个字节
def to_little_end_byte_list(val, byte_cnt=-1):
    byte_list = []
    if type(val) == int:
        while val > 0 and byte_cnt > 0:
            byte_list.append(val % 256)
            val //= 256
            byte_cnt -= 1
        while byte_cnt > 0:
            byte_list.append(0)
            byte_cnt -= 1
    elif type(val) == str: # 所赋值为字符串
        byte_list = [ord(c) for c in val]
        byte_list.append(0x0) # 作为字符串
    return byte_list

# 添加导出表
def get_modifiable_range_in_exports(fn, functions_cnt=0x10, write_path="tmp/tmp_exe.exe"):
    exe_info = lief.parse(fn)
    mrl = []
    s5 = lief.PE.Section()

    s5.name = 'zbh'
    s5.virtual_size = 0x1111
    s5.sizeof_raw_data = 0x2000
    s5.virtual_address
    head_content = [
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,  # 'name'
        0x01, 0x00, 0x00, 0x00,  # 'base'
        0x00, 0x00, 0x00, 0x00,  # 'numberOfFunctions'
        0x00, 0x00, 0x00, 0x00,  # 'numberOfNames'
        0x00, 0x00, 0x00, 0x00,  # 'addressOfFunctions'
        0x00, 0x00, 0x00, 0x00,  # 'addressOfNames'
        0x00, 0x00, 0x00, 0x00,  # 'addressOfNameOrdinals'
    ]
    content = [0x0] * s5.size
    content[0:40] = head_content

    # 向文件添加新的节
    exe_info.add_section(s5)
    es = exe_info.sections[len(exe_info.sections) - 1]
    # es.virtual_address

    # 修改导出表的地址和大小
    expt_info = exe_info.data_directories[0]
    expt_info.rva = es.virtual_address
    expt_info.size = es.size

    after_section_head_rva = es.virtual_address + 0x40

    # functions_cnt = 10
    function_names_cnt = functions_cnt
    functions_t_rva = after_section_head_rva
    function_name_ordinals_rva = functions_t_rva + functions_cnt * 4
    function_name_addr_rva = function_name_ordinals_rva + functions_cnt * 2  # 函数名字符串的指针数组
    function_names_rva = function_name_addr_rva + functions_cnt * 4  # 把函数名字符串放最后面

    functions_t_offset = 0x40
    functions_name_ordinals_offset = 0x40 + functions_cnt * 4
    functions_name_addr_offset = functions_name_ordinals_offset + functions_cnt * 2
    functions_name_offset = functions_name_addr_offset + functions_cnt * 4

    content[20: 24] = to_little_end_byte_list(functions_cnt, 4)  # [0xf, 0xf, 0xf, 0xf, ]
    content[24: 28] = to_little_end_byte_list(function_names_cnt, 4)
    content[28: 32] = to_little_end_byte_list(functions_t_rva, 4)
    content[32: 36] = to_little_end_byte_list(function_name_addr_rva, 4)
    content[36: 40] = to_little_end_byte_list(function_name_ordinals_rva, 4)

    offset = functions_t_offset
    for i in range(functions_cnt):
        content[offset:offset + 4] = to_little_end_byte_list(0x1000, 4)
        offset += 4

    offset = functions_name_ordinals_offset
    for i in range(function_names_cnt):
        content[offset: offset + 2] = to_little_end_byte_list(i, 2)
        offset += 2

    offset = functions_name_addr_offset
    for i in range(function_names_cnt):
        content[offset: offset + 4] = to_little_end_byte_list(function_names_rva, 4)
        function_names_rva += 8
        offset += 4

    offset = functions_name_offset
    section_offset = es.offset
    for i in range(function_names_cnt):
        content[offset: offset + 8] = to_little_end_byte_list(
            ''.join(random.sample('zyxwvutsrqponmlkjihgfedcba9876543210_',7)),
            7
        )
        mrl.append((section_offset + offset, section_offset + offset + 7))
        offset += 8

    es.content = content

    bld = lief.PE.Builder(exe_info)
    bld.build()
    bld.write(write_path)

    # bytez = open(write_path, 'rb').read()
    # byte_ary = list(struct.unpack('B' * len(bytez), bytez))
    return mrl #, [byte_ary]

def get_modifiable_range_in_section_table(fn, new_sections_cnt=0x10, write_path="tmp/tmp_exe.exe"):
    exe_info = lief.parse(fn)
    bytez = open(fn, 'rb').read()
    byte_ary = list(struct.unpack('B' * len(bytez), bytez))
    mrl = [] # 可改字节的位置范围存于此数组

    e_lfanew = byte_ary[0x3c: 0x40]
    e_lfanew.reverse()

    optional_header_offset = reduce(lambda x, y: x * 256 + y, e_lfanew)
    section_table_offset = optional_header_offset + 4 + 20 + exe_info.header.sizeof_optional_header
    offset = section_table_offset

    # 创建新节
    for _ in range(new_sections_cnt):
        extra_s = lief.PE.Section()
        extra_s.name = ''.join(random.sample('zyxwvutsrqponmlkjihgfedcba9876543210_', 7))
        extra_s.virtual_size = 0x10
        extra_s.size = extra_s.virtual_size
        extra_s.characteristics = 0x40000040
        # extra_s.virtual_address = 0x1000
        extra_s.content = [0xff] * extra_s.size
        exe_info.add_section(extra_s)

        mrl.extend([
            (offset, offset + 7), # 节名
            (offset + 8 + 8, offset + 8 + 8 + 3), # SizeOfRawData 前3字节
            (offset + 36, offset + 36 + 4), # Characteristics
        ])
        offset += 40

    bld = lief.PE.Builder(exe_info)
    bld.build()
    bld.write(write_path)

    return mrl

def find_pe_modifiable_range_ember_preproc(fn, export_func_cnt=10, inserted_sec_cnt=0x8, write_path="tmp/tmp_exe.exe"):
    mrl1 = find_pe_modifiable_range(fn, use_range=0b10) # 取PE头部可写字段的字节位置
    mrl2 = get_modifiable_range_in_exports(fn, export_func_cnt, write_path) # 插入导出表, 取出导出函数名的字节位置
    mrl3 = get_modifiable_range_in_section_table(write_path, inserted_sec_cnt, write_path) # 插入节, 取出节名等的字节位置
    mrl = mrl1 + mrl2 + mrl3
    bytez = open(write_path, 'rb').read()
    byte_ary = list(struct.unpack('B' * len(bytez), bytez))
    return mrl, [byte_ary]

def get_pe_info(fn_list):
    pass