import json
import os
import xml.etree.ElementTree as ET
import xlwt
from openpyxl import load_workbook


# 解析json
def parse_json(config_path):
    if os.path.isfile(config_path) and config_path.endswith('json'):
        data = json.load(open(config_path))
        data = data['data']
        return data


# 解析Annotation文件
def parse_annotation(xml_path, category_id_and_name):
    in_file = open(xml_path)
    tree = ET.parse(in_file)
    root = tree.getroot()
    boxes = []
    category_ids = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text

        if category_id_and_name.get(cls) != None:
            cls_id = category_id_and_name[cls]
            xmlbox = obj.find('bndbox')
            boxes.append([int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)])
            category_ids.append(cls_id)

    return boxes, category_ids


def modify_json(config_path):
    if os.path.isfile(config_path) and config_path.endswith('json'):
        with open(config_path, "r+") as jsonFile:
            json_data = json.load(jsonFile)
            data = json_data['data']
            data = data['train']["pipeline"]
            for aug in data:
                if "type" in aug.keys():
                    if aug["type"] == "Normalize":
                        aug["mean"] = [127.888, 134.39, 135.685]
                        aug["std"] = [64.839, 68.405, 67.173]
            jsonFile.seek(0)  # rewind
            json.dump(data, jsonFile,ensure_ascii=False)
            jsonFile.truncate()


def write_excel(augment_infos, save_path, sheet_name):
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet = workbook.add_sheet(sheet_name)

    # 写入excel
    # 参数对应 行, 列, 值
    dict_index = augment_infos[0]

    for i, data in enumerate(augment_infos[1:]):
        for j, key in enumerate(data):
            value = data[key]
            if isinstance(value,str) == False:
                value = str(value)
            worksheet.write(i+1, dict_index[key], label=value)
    # 保存
    workbook.save(save_path)


if __name__ == "__main__":
    path = "test.xlsx"
    json_paths = "../configs"
    json_list = os.listdir(json_paths)

    augment_infos_list = [{"Augmentation":0, "pram":5, "acc":4, "size":2, "epoch":3, "model":1}]
    for json_path in json_list:
        augment_name = json_path[:-5]
        json_path = os.path.join(json_paths, json_path)

        excel_info = {"pram":"", "Augmentation":""}
        augments_info = parse_json(json_path)['train']['pipeline']
        augments_info_txt = "\n%s:\n" % (augment_name)
        for augment_info in augments_info:
            if "type" in augment_info.keys():
                for key,value in augment_info.items():
                    pram_info = "%s:%s\n" % (key, value)
                    augments_info_txt += pram_info
                    excel_info["pram"] += pram_info
                    if value == "Resize" and key == "type":
                        excel_info["size"] = augment_info["size"]

                    if (value != "Resize" and value != "Normalize") and key == "type":
                        excel_info["Augmentation"] = value

                augments_info_txt += "\n"
        print(augments_info_txt)

        excel_info["model"] = "MobileNetV3"
        excel_info["epoch"] = 40
        excel_info["acc"] = 0.9888
        augment_infos_list.append(excel_info)
        write_excel(augment_infos_list, save_path=path, sheet_name="MobileNetV3")
        pass


