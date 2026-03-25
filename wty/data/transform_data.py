import os
import cv2
import json
from tqdm import tqdm
with open('/hdd/wangty/new_task/cervical_vertebra/dataset/bbox/tra_box_fill.json',"r") as f:
    tra_box=json.load(f)
with open('/hdd/wangty/new_task/cervical_vertebra/dataset/bbox/foraminal_l.json',"r") as f:
    f_l=json.load(f)
with open('/hdd/wangty/new_task/cervical_vertebra/dataset/bbox/foraminal_r.json',"r") as f:
    f_r=json.load(f)
img_folder='/hdd/wangty/new_task/images/origin_png/tra'
new_folder='/mnt/nvme_share/wangty/img/tra_512'
box_folder='/home/wangty/github/emu3/Emu3/wty/data/box512'
import os
import cv2
import json
from tqdm import tqdm

# ================= 1. 配置路径和初始数据 =================
# img_folder = '/hdd/wangty/new_task/images/origin_png/tra'
# new_folder = '/hdd/wangty/new_task/images/processed_png/tra'
# box_folder = '/hdd/wangty/new_task/boxes'

target_size = 512

# 根据你的描述，9个索引对应的文件名
file_names = ['3.png', '3-4.png', '4.png', '4-5.png', '5.png', '5-6.png', '6.png', '6-7.png', '7.png']

# 这里假设你已经在内存中有了这三个字典数据
# tra_box = {'01240515218048': [[230, 262, 285, 297], ...], ...}
# f_l = {...}
# f_r = {...}

# 初始化新的定位框字典用于保存
new_tra_box = {}
new_f_l = {}
new_f_r = {}

# ================= 2. 定义核心转换函数 =================
def transform_box(box, w_orig, h_orig, target_size=512):
    """同步更新定位框坐标 [xmin, ymin, xmax, ymax]"""
    if not box or len(box) != 4:
        return box
    
    x1, y1, x2, y2 = box
    short_edge = min(w_orig, h_orig)

    # 1. 中心裁剪带来的平移偏移量
    left_offset = (w_orig - short_edge) // 2
    top_offset = (h_orig - short_edge) // 2

    x1_crop = x1 - left_offset
    y1_crop = y1 - top_offset
    x2_crop = x2 - left_offset
    y2_crop = y2 - top_offset

    # 边界保护：限制在裁剪后的正方形范围内
    x1_crop = max(0, min(x1_crop, short_edge))
    y1_crop = max(0, min(y1_crop, short_edge))
    x2_crop = max(0, min(x2_crop, short_edge))
    y2_crop = max(0, min(y2_crop, short_edge))

    # 2. 缩放带来的比例映射
    scale = target_size / short_edge
    x1_new = int(round(x1_crop * scale))
    y1_new = int(round(y1_crop * scale))
    x2_new = int(round(x2_crop * scale))
    y2_new = int(round(y2_crop * scale))

    return [x1_new, y1_new, x2_new, y2_new]

def process_image(img, target_size=512):
    """图像裁剪与缩放，返回处理后的图像及原始宽高"""
    h_orig, w_orig = img.shape[:2]
    short_edge = min(w_orig, h_orig)

    # 1. 中心裁剪
    left_offset = (w_orig - short_edge) // 2
    top_offset = (h_orig - short_edge) // 2
    img_cropped = img[top_offset:top_offset + short_edge, left_offset:left_offset + short_edge]

    # 2. Resize 最佳插值算法选择
    # 缩小用 INTER_AREA，放大用 INTER_CUBIC
    interp = cv2.INTER_AREA if short_edge > target_size else cv2.INTER_CUBIC
    img_resized = cv2.resize(img_cropped, (target_size, target_size), interpolation=interp)
    
    return img_resized, w_orig, h_orig

# ================= 3. 执行主循环 =================
os.makedirs(new_folder, exist_ok=True)
os.makedirs(box_folder, exist_ok=True)

# 遍历字典中的每一个患者 ID（假设三个字典的 keys 是一致的）
for patient_id in tqdm(tra_box.keys()):
    # 为每组 ID 创建对应的保存目录
    patient_new_folder = os.path.join(new_folder, str(patient_id))
    os.makedirs(patient_new_folder, exist_ok=True)
    
    # 初始化当前 ID 在三个新字典中的列表
    new_tra_box[patient_id] = []
    new_f_l[patient_id] = []
    new_f_r[patient_id] = []
    
    for i in range(9):
        file_name = file_names[i]
        img_path = os.path.join(img_folder, str(patient_id), file_name)
        new_img_path = os.path.join(patient_new_folder, file_name)
        
        # 读取原图
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: 无法读取图像 {img_path}")
            # 如果图像缺失，你可以选择填入空值或保持原样
            new_tra_box[patient_id].append([])
            new_f_l[patient_id].append([])
            new_f_r[patient_id].append([])
            continue
            
        # 处理图像
        img_resized, w_orig, h_orig = process_image(img, target_size)
        
        # 保存新图像
        cv2.imwrite(new_img_path, img_resized)
        
        # 同步更新并保存对应的 3 个框
        # 注意需要判断是否存在对应索引的框，避免越界
        box_tra = tra_box[patient_id][i] if i < len(tra_box[patient_id]) else []
        box_fl = f_l[patient_id][i] if i < len(f_l[patient_id]) else []
        box_fr = f_r[patient_id][i] if i < len(f_r[patient_id]) else []
        
        new_tra_box[patient_id].append(transform_box(box_tra, w_orig, h_orig, target_size))
        new_f_l[patient_id].append(transform_box(box_fl, w_orig, h_orig, target_size))
        new_f_r[patient_id].append(transform_box(box_fr, w_orig, h_orig, target_size))

# ================= 4. 保存为 JSON 格式 =================
with open(os.path.join(box_folder, 'tra_box.json'), 'w', encoding='utf-8') as f:
    json.dump(new_tra_box, f, ensure_ascii=False, indent=4)

with open(os.path.join(box_folder, 'f_l.json'), 'w', encoding='utf-8') as f:
    json.dump(new_f_l, f, ensure_ascii=False, indent=4)

with open(os.path.join(box_folder, 'f_r.json'), 'w', encoding='utf-8') as f:
    json.dump(new_f_r, f, ensure_ascii=False, indent=4)

print("数据预处理完成！图像与定位框已同步更新并保存。")