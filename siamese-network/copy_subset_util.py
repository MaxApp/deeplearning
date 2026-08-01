import os
import shutil
import re
from collections import defaultdict


def create_signature_subset(
    src_dir='full_org',
    dst_dir='new_full_org',
    num_per_user=3,
    ext='.png'
):
    """
    从 src_dir 中，按用户 ID 提取每个用户编号前 num_per_user 张图片
    
    文件名格式: <prefix>_<user_id>_<img_num>.png
    示例: forgeries_1_2.png, original_1_2.png
    """
    # 正则匹配: prefix_userid_imgnum.ext
    pattern = re.compile(r'^(.+?)_(\d+)_(\d+)' + re.escape(ext) + r'$', re.IGNORECASE)
    
    # 收集文件: key=(relative_dir, user_id)
    user_files = defaultdict(list)
    
    for root, dirs, files in os.walk(src_dir):
        rel_dir = os.path.relpath(root, src_dir)
        for filename in files:
            if not filename.lower().endswith(ext.lower()):
                continue
            match = pattern.match(filename)
            if not match:
                continue
            prefix = match.group(1)
            user_id = int(match.group(2))
            img_num = int(match.group(3))
            
            src_path = os.path.join(root, filename)
            user_files[(rel_dir, user_id)].append({
                'filename': filename,
                'prefix': prefix,
                'img_num': img_num,
                'src_path': src_path,
                'rel_dir': rel_dir
            })
    
    if not user_files:
        print(f"未在 {src_dir} 中找到匹配的文件（格式: *_*_*{ext}）")
        return
    
    copied_total = 0
    user_count = 0
    insufficient_users = []
    
    for (rel_dir, user_id), files in sorted(user_files.items()):
        # 按图片编号排序，编号相同则按前缀排序
        files_sorted = sorted(files, key=lambda x: (x['img_num'], x['prefix']))
        selected = files_sorted[:num_per_user]
        
        if len(files_sorted) < num_per_user:
            insufficient_users.append(
                f"  User {user_id} (目录: '{rel_dir}'): "
                f"仅有 {len(files_sorted)} 张，已复制全部"
            )
        
        for item in selected:
            dst_path_dir = os.path.join(dst_dir, rel_dir)
            os.makedirs(dst_path_dir, exist_ok=True)
            dst_path = os.path.join(dst_path_dir, item['filename'])
            shutil.copy2(item['src_path'], dst_path)
            copied_total += 1
        
        user_count += 1
    
    print(f"完成！共处理 {user_count} 个用户，复制 {copied_total} 张图片到 {dst_dir}")
    if insufficient_users:
        print(f"\n以下用户图片不足 {num_per_user} 张：")
        for msg in insufficient_users:
            print(msg)

if __name__ == '__main__':
    create_signature_subset()