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


def create_signature_map(base_data_dir):
        """
        read dataset directory and use a dict to organize the signature files by userid,
        with 'real' and 'fake' part.
        """
        from collections import defaultdict
        import glob
        real_signatures_dir = os.path.join(base_data_dir, 'DigitalReal')
        fake_signatures_dir = os.path.join(base_data_dir, 'DigitalFake')
        signature_map = defaultdict(lambda: {'real': [], 'fake': []})
        if not os.path.isdir(real_signatures_dir):
            raise FileNotFoundError(f"Error: Directory not found at {real_signatures_dir}")
        if not os.path.isdir(fake_signatures_dir):
            raise FileNotFoundError(f"Error: Directory not found at {fake_signatures_dir}")
        all_filenames = sorted(os.listdir(real_signatures_dir))
        uid = "0"
        for i, filename in enumerate(all_filenames):
            # print(f"filename: {filename}")
            match = re.search(r'digital_[a-z]+_(\d+)_(\d+)', filename)
            if match:
                tmp_uid = match.group(1)
                # print(f"tmp_uid: {tmp_uid}")
                if tmp_uid != uid:
                    uid = tmp_uid
                    file_person_real = os.path.join(real_signatures_dir , 'digital_real_' + uid + '_' + '*.jpg')
                    file_person_fake = os.path.join(fake_signatures_dir , 'digital_fake_' + uid + '_' + '*.jpg')
                    real_images = glob.glob(file_person_real)
                    fake_images = glob.glob(file_person_fake)
                    # print(f"real: {real_images}")
                    # print(f"fake: {fake_images}")

                    # filter the satisfied dataset to make triplet
                    if len(real_images) >= 2 and len(fake_images) >= 1:
                        signature_map[int(uid)]['real'] = real_images
                        signature_map[int(uid)]['fake'] = fake_images
                #     else:
                #         print("2"*10)
                # else:
                #     print("1111")
            # else:
            #     print(f"match nothing")

            # if i>10:
            #     break
        return signature_map

if __name__ == "__main__":
    map = create_signature_map("E:\\PDF\\pytorch\\C3M1\\signatures_train")
    print(f"total: {len(map)}")
    for k,v in sorted(map.items()):
        print(f"uid: {k}  real: {len(v['real'])}  fake: {len(v['fake'])}")

    print(f"total: {len(map)}")
    # print(os.path.join(os.path.join('a','b'), 'c'+'d'))