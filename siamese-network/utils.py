import os
import re


def rename_files(file_dir:str):
    if not file_dir:
        return
    
    filenames = os.listdir(file_dir)
    id = "0"
    count = 1
    for i,name in enumerate(filenames):
        # if i>10:
        #     break
        print(f"{name}")
        match = re.search(r'digital_[a-z]+_(\d+)_', name)
        if match:
            name_id = match.group(1)
            if name_id != id:
                id = name_id
                count = 1
            else:
                count += 1

            start_idx = name.find("_png")
            end_idx = name.find(".jpg")
            if start_idx == -1 or end_idx == -1:
                continue

            new_name = name[0:start_idx] + "_" + str(count) + name[end_idx:] 
            print(f"{new_name}")

            os.rename(os.path.join(file_dir, name), os.path.join(file_dir, new_name))


if __name__ == "__main__":
    rename_files("")