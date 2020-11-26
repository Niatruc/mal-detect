import lief

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

        # MajorOperatingSystemVersion(2), MinorOperatingSystemVersion(2), MajorImageVersion(2), MinorImageVersion(2)
        pe_header_modifiable_range3 = (image_optional_header_offset + 2 + 14 + 24, image_optional_header_offset + 2 + 14 + 24 + 8)

        # Win32VersionValue(4)
        pe_header_modifiable_range4 = (image_optional_header_offset + 2 + 14 + 24 + 8 + 4, image_optional_header_offset + 2 + 14 + 24 + 8 + 4 + 4)

        checksum_offset = image_optional_header_offset + 2 + 14 + 24 + 8 + 4 + 4 + 8

        # CheckSum(4)
        pe_header_modifiable_range5 = (checksum_offset, checksum_offset + 4)

        # LoaderFlags(4)
        pe_header_modifiable_range6 = (checksum_offset + 4 + 20, checksum_offset + 4 + 20 + 4)

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
            [dos_header_modifiable_range1, dos_header_modifiable_range2],
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