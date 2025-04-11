import random
import copy
import datetime
import pickle
import os
import re
import time
import torch
import sys
import csv
import traceback
import numpy as np

store_no = sys.argv[-1]

def seed_everything(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()

def make_test_dataset(menu_phrase, option_phrase):
    template_suffix = random.choice(intent_suffix_template['place_order'])
    full_sentence = menu_phrase + ' ' + option_phrase + ' ' + template_suffix    
    full_sentence = full_sentence.strip()
    full_sentence = ' '.join(full_sentence.split())
        
    return full_sentence

def remove_special_char(prod_name):
    modified_prod_name = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", ' ', prod_name)
    modified_prod_name = modified_prod_name.strip()
    return modified_prod_name


def make_templates(generated_dataset_list, menu_phrase, option_group_name=None, option_phrase=None, use_option=False, for_test=False):
    full_sentence = ''
    
    # ? check option usage
    if use_option:
        template_list = food_option_template_list
        # @ option phrase
        if isinstance(option_phrase, str) and option_group_name is not None:
            if random.randint(0, 1):
                option_phrase = f'{option_group_name}(으)로 {option_phrase}'

    else:
        template_list = random.choice(food_template_list)

    # @ select quantity
    qty_phrase = random.choice(qty_phrase_list)

    # select template suffix
    template_suffix = random.choice(intent_suffix_template['place_order'])

    # select template order
    template_order = copy.deepcopy(template_list)

    # shuffle template order
    random.shuffle(template_order)

    # make sentence by template order
    if use_option and isinstance(option_phrase, list):
        for opt_prs in option_phrase:
            full_sentence = ''
            for n, ent in enumerate(template_order):
                if ent == 'food':
                    full_sentence += menu_phrase
                    if use_option and n != (len(template_order) - 1) and template_order[n+1]=='option':
                        full_sentence += '에'
                elif ent == 'option':
                        full_sentence += opt_prs
                elif ent == 'qty':
                    full_sentence += qty_phrase
                else:
                    raise NotImplementedError

                full_sentence += ' '

            # add suffix
            full_sentence += template_suffix
            full_sentence = full_sentence.strip()

            # add to list
            generated_dataset_list.append(full_sentence)

    # only option phrase (without menu)
    elif menu_phrase is None:
        full_sentence += option_phrase + ' '
        full_sentence += template_suffix
        full_sentence = full_sentence.strip()

        # add to list
        generated_dataset_list.append(full_sentence)

    else:
        for n, ent in enumerate(template_order):
            if ent == 'food':
                full_sentence += menu_phrase
                if use_option and n != (len(template_order) - 1) and template_order[n+1]=='option':
                    full_sentence += '에'
            elif ent == 'option':
                full_sentence += option_phrase
            elif ent == 'qty':
                full_sentence += qty_phrase
            else:
                raise NotImplementedError

            full_sentence += ' '

        # add suffix
        full_sentence += template_suffix
        full_sentence = full_sentence.strip()

        # add to list
        generated_dataset_list.append(full_sentence)
        
    return generated_dataset_list

def convert_num_to_char(prod_name):
    hangle_num=[['하나', '한', '일'], ['둘', '두', '이'], ['셋', '세', '새', '석', '삼'], 
            ['넷', '네', '내', '사', '넉'], ['다섯', '닷', '오'],
              ['여섯', '엿', '육'], ['일곱', '칠'], ['여덟', '여덜', '팔'], ['아홉', '압', '구']]
    """_summary_

    Args:
        prod_name (_type_): 전체 메뉴명

    Returns:
        _type_: _description_
        숫자를 다 텍스트로 바꾸기
    """
    reg = re.compile(r'[a-zA-Z]')
    count_int = 0
    for i in prod_name:
        if i.isdigit():
            count_int += 1
    if count_int == 0 :
        return [prod_name]
    
    total_converted_list = list()
    for i in range(count_int):
        if i == 0:
            for n, char in enumerate(prod_name):
                if n < len(prod_name) - 1:
                    if char.isdigit() and prod_name[n+1].isalpha() and reg.findall(prod_name[n+1]) == []:
                        # 해당 글자 다음이 숫자일 경우 제외 (10 이상 숫자는 변환하지 않음) & 다음 글자가 영어일 경우 제외 (1L 이런 경우)
                        if 0 < int(char) <= 9 :
                            hangle_list = hangle_num[int(char)-1]
                            for hangle in hangle_list:
                                temp_prod_name = prod_name
                                temp_prod_name = temp_prod_name[:n] + ' ' + hangle + ' ' + temp_prod_name[n+1:]
                                total_converted_list.append(temp_prod_name)
        else :
            for m, converted_text in enumerate(total_converted_list):
                for n, char in enumerate(converted_text):
                    if n < len(converted_text) - 1:
                        if char.isdigit() and converted_text[n+1].isalpha() and reg.findall(converted_text[n+1]) == []:
                            # 해당 글자 다음이 숫자일 경우 제외 (10 이상 숫자는 변환하지 않음)
                            if 0 < int(char) <= 9 :
                                hangle_list = hangle_num[int(char)-1]
                                for hangle in hangle_list:
                                    temp_prod_name = converted_text
                                    temp_prod_name = temp_prod_name[:n] + ' ' + hangle + ' ' + temp_prod_name[n+1:]
                                    total_converted_list[m] = temp_prod_name    
    
    if total_converted_list == []:
        total_converted_list = [prod_name]
                    
    return total_converted_list

def convert_temp_to_str(name):
    term_list = list()
    menu_hot_list = [['hot'], ['핫', '뜨거운', '따뜻한']]
    menu_ice_list = [['iced', 'ice'], ['아이스', '콜드', '차가운', '찬', '시원한']]
    converted = False
    
    for menu_term_list in [menu_hot_list, menu_ice_list]:
        for term in menu_term_list[0]:
            lower_name = name.lower()
            if term in lower_name:
                converted = True
                for candidate_term in menu_term_list[1]:
                    term_list.append(lower_name.replace(term, candidate_term))
            if converted:
                break
    
    return term_list   


def convert_alphabet_to_kor(converted_name_list: list):
    ENGS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    KORS = ['에이', '비', '씨', '디', '이', '에프', '지', '에이치', '아이', '제이',
            '케이', '엘', '엠', '엔', '오', '피', '큐', '알', '에스', '티', 
            '유', '브이', '더블유', '엑스', '와이', '지']

    eng2kor = dict(zip(ENGS, KORS))
    
    is_english = re.compile(r'[A-Z]') # 영어 정규화
    
    converted_korean_list = []
    for name in converted_name_list:
        temp = is_english.findall(name) # 함수에 들어오는 인자가 영어가 있는경우 temp에 삽입

        if len(temp) > 0: # 영어가 temp에 존재하는 경우
            # result_trans.append([prod, [eng2kor[i] for i in temp if i.isupper()]])
            for i in temp:
                if i.isupper():
                    name = name.replace(i, eng2kor[i])
                
            converted_korean_list.append(name)

    converted_name_list.extend(converted_korean_list)
        
    return converted_name_list

def prepro_name(name):
    cnt = 0
    name_list = list()
    
    # 원본 이름 그대로 사용
    name_list.append(name)
    
    # ! 숫자 -> 한글 변환 
    converted_name_list = convert_num_to_char(name)    
    
    # $ 11/26 알파벳 -> 한글 추가
    converted_name_list = convert_alphabet_to_kor(converted_name_list)

    # ! 특수문자 제거
    for name in converted_name_list:
        name = name.replace('(', ' ').replace(')', '')
        total_replace_list = list()
        if '%' in name:
            special_char = '%'
            replace_list = ['퍼센트', '퍼', '']
            total_replace_list.append([special_char, replace_list])
        if '.' in name:
            special_char = '.'
            replace_list = ['점']
            total_replace_list.append([special_char, replace_list])
        if '/' in name:
            special_char = '/'
            
            # 분리해서 만들기
            split_list = name.split(' ')
            remake_list = list()
            remake_template = ''
            for t in split_list:
                if '/' in t:
                    slash_split_list = t.split('/')
                    # 물/우유 -> 물, 우유
                    remake_list = [slash_text for slash_text in slash_split_list]
                else :
                    if remake_list == []:
                        remake_template += ' ' + t
                    else :
                        remake_list = [remake_template + slash_text + ' ' + t for slash_text in slash_split_list]
                        
            name_list.extend(remake_list)
                
            # / 치환
            replace_list = [' 또는 ', '(이)나 ']
            total_replace_list.append([special_char, replace_list])
            
        if '&' in name:
            special_char = '&'
            replace_list = ['와(과) ', '(이)랑 ', '에 ', ' 엔 ', ' 앤 ', ' ']
            total_replace_list.append([special_char, replace_list])
        if '+' in name:
            special_char = '+'
            replace_list = ['와(과) ', '(이)랑 ', '에 ', ' ']
            total_replace_list.append([special_char, replace_list])
        if ':' in name:
            special_char = ':'
            replace_list = ['와(과) ', '(이)랑 ', '에 ', ' ']
            total_replace_list.append([special_char, replace_list])
            
        if total_replace_list != [] :
            for special_char, replace_text in total_replace_list:
                for replace_text in replace_list:
                    cnt+=1
                    name_list.append(name.replace(special_char, replace_text))
        name_list.append(name)
    
    # ! 띄어쓰기 임의 추가
    if len(name) > 1:
        tmp_name_list = copy.deepcopy(name_list)
        for name in tmp_name_list:
            tmp_name = name
            len_of_name = len(name)
            for _ in range(len_of_name):
                if len_of_name >= 2:
                    random_cnt = random.randint(1, len_of_name//2)
                    for _ in range(random_cnt):
                        random_idx = random.randint(1, len(name) - 1)
                        tmp_name = name[:random_idx] + ' ' + name[random_idx:]
                        tmp_name = tmp_name.replace('  ', ' ')
                
                name_list.append(tmp_name)
        
        
    # 중복 제거 
    name_list = list(set(name_list))
        
    return name_list

if __name__ == '__main__':
    try: 
        start_time = time.time()
        print(time.ctime())

        data_path = './data/trained'

        intent_template_path = 'utils/template_per_intents.csv' 
        with open(intent_template_path, 'r', encoding='cp949') as f:
            intent_template = f.readlines()    
            
        intent_suffix_template = dict()
        qty_phrase_list = list()
        qty_phrase_beverage_list = list()

        for n, line in enumerate(intent_template):
            if n == 0 :
                continue
            line_list = line.split(',') # idx 3이 qty_phrase, 4가 qty_phrase_beverage
            if line_list[0] not in intent_suffix_template.keys() and line_list[0] != '':
                intent_suffix_template[line_list[0]] = list()
            if line_list[2] != '':
                intent_suffix_template[line_list[0]].append(line_list[2])
            if line_list[3] != '' and line_list[4] != '':
                qty_phrase_list.append(line_list[3])
                if line_list[4].strip() != '':
                    qty_phrase_beverage_list.append(line_list[4].strip())

        food_template_list = [['food', 'qty'], ['food']]
        food_option_template_list = ['food', 'option', 'qty']

        
        store_path = os.path.join(data_path, store_no)
        prod_opt_path = os.path.join(store_path, 'prod-opt_dict.pkl')
        with open(prod_opt_path, 'rb') as f:
            prod_opt_dict = pickle.load(f)   
        # store_category_dict = store_dict[store]
        
        # ! 주문 발화문 전체 데이터셋 리스트
        place_order_generated_dataset_list = list()
        
        # ? TEST 발화문 전체 데이터셋 리스트
        place_order_generated_test_dataset_list = list()
        
        # $ fallback 발화문 전체 데이터셋 리스트
        fallback_generated_dataset_list = list()
        
        # -- 취소 발화문 전체 데이터셋 리스트
        menu_cancel_generated_dataset_list = list()
        
        # ? 슬롯 별 개수 확인 
        cnt_per_slot = dict()
        after_cnt_per_slot = dict()
        
        for menu_code, menu_dict in prod_opt_dict.items():
            cnt_per_slot[menu_code] = {}
            cnt_per_slot[menu_code]['cnt'] = 0
            after_cnt_per_slot[menu_code] = {}
            after_cnt_per_slot[menu_code]['cnt'] = 0
            # category_name = menu_dict['category_name']
            
            # 카테고리 타입 -> drink, food 중 하나
            # category_type = store_category_dict[category_name]
            
            # 메뉴 이름
            menu_name = menu_dict['menu_name'].replace('<','').replace('>','').replace(':','')
            ori_menu_name = menu_name
            cnt_per_slot[menu_code]['name'] = ori_menu_name
            ori_menu_phrase = f'<{ori_menu_name}:{menu_code}>'
            
            # 메뉴 이름 전처리
            menu_name_list = prepro_name(menu_name)
            
            # 옵션
            options = menu_dict['options']
            
            # ? TEST DATASET
            clean_menu_name = remove_special_char(ori_menu_name)
            clean_menu_phrase = f'<{clean_menu_name}:{menu_code}>'
            
            if options is None:
                place_order_generated_test_dataset_list.append(make_test_dataset(clean_menu_phrase, ''))
                
            else:
                for option_code, option_info in options.items():
                    option_name, option_group_name = option_info['option_name'], option_info['option_group_name']
                    ori_option_name = option_name
                    ori_option_phrase = f'<{ori_option_name}:{option_code}>'
                    
                    clean_option_name = remove_special_char(ori_option_name)
                    clean_option_phrase = f'<{clean_option_name}:{option_code}>'
                    place_order_generated_test_dataset_list.append(make_test_dataset(clean_menu_phrase, clean_option_phrase))
            
            for menu_name in menu_name_list:
            
                # @ 메뉴 이름 태깅
                menu_phrase = f'<{menu_name}:{menu_code}>'
                
                # $ fallback
                if random.randint(0, 1):
                    # if original menu name includes integer, skip making fallback phrase
                    if not bool(re.search(r'\d', ori_menu_name)):
                        fallback_suffix = random.choice(intent_suffix_template['fallback'])
                        fallback_sentence = f'{menu_phrase} {fallback_suffix}'
                        fallback_generated_dataset_list.append(fallback_sentence)
                
                # -- 취소
                if random.randint(0, 1):
                    # if original menu name includes integer, skip making cancel phrase
                    if not bool(re.search(r'\d', ori_menu_name)):
                        menu_cancel_suffix = random.choice(intent_suffix_template['menu_cancel'])
                        menu_cancel_sentence = f'{menu_phrase} {menu_cancel_suffix}'
                        menu_cancel_generated_dataset_list.append(menu_cancel_sentence)
                
                # ! 메뉴명만
                place_order_generated_dataset_list.append(menu_phrase)
                
                # @ ICE, HOT
                converted_list = convert_temp_to_str(ori_menu_phrase)
                if converted_list != []:
                    place_order_generated_dataset_list.extend(convert_temp_to_str(ori_menu_phrase))
                
                # ! 옵션 없이 메뉴명 & 수량
                place_order_generated_dataset_list = make_templates(place_order_generated_dataset_list, menu_phrase, option_group_name=None, option_phrase=None, use_option=False)
                
                if options is None:
                    continue
                
                # ! 완전한 주문발화 - 옵션 하나씩
                for option_code, option_info in options.items():
                    cnt_per_slot[option_code] = {}
                    cnt_per_slot[option_code]['cnt'] = 0
                    after_cnt_per_slot[option_code] = {}
                    after_cnt_per_slot[option_code]['cnt'] = 0
                    
                    option_name, option_group_name = option_info['option_name'], option_info['option_group_name']
                    ori_option_name = option_name
                    cnt_per_slot[option_code]['name'] = ori_option_name
                    ori_option_phrase = f'<{ori_option_name}:{option_code}>'

                    option_name_list = prepro_name(option_name)
                    
                    # 옵션은 한 번만
                    option_name = random.choice(option_name_list)
                    # for option_name in option_name_list:
                    
                    # @ 옵션 구문
                    option_phrase = f'<{option_name}:{option_code}>'
                    
                    # ! 옵션명만 
                    place_order_generated_dataset_list.append(option_phrase)
                    if option_group_name != '[NULL]':
                        place_order_generated_dataset_list.append(f'{option_group_name}(으)로 {option_phrase}')
                    else:
                        option_group_name=None
                    
                    # @ ICE, HOT
                    place_order_generated_dataset_list.extend(convert_temp_to_str(ori_option_phrase))
                    converted_temp_str_phrase = convert_temp_to_str(ori_option_phrase)
                    place_order_generated_dataset_list = make_templates(place_order_generated_dataset_list, menu_phrase, option_group_name, converted_temp_str_phrase, use_option=True)
                    
                    # for _ in range(3):
                    place_order_generated_dataset_list = make_templates(place_order_generated_dataset_list, menu_phrase, option_group_name, option_phrase, use_option=True)
                    # 옵션그룹명 없이
                    place_order_generated_dataset_list = make_templates(place_order_generated_dataset_list, menu_phrase, option_group_name=None, option_phrase=option_phrase, use_option=True)        
                
            place_order_generated_dataset_list = [' '.join(line.split()) for line in place_order_generated_dataset_list]
            fallback_generated_dataset_list = [' '.join(line.split()) for line in fallback_generated_dataset_list]
            menu_cancel_generated_dataset_list = [' '.join(line.split()) for line in menu_cancel_generated_dataset_list]
            
            place_order_generated_dataset_list = list(set(place_order_generated_dataset_list))
            fallback_generated_dataset_list = list(set(fallback_generated_dataset_list))
            menu_cancel_generated_dataset_list = list(set(menu_cancel_generated_dataset_list))
            
            place_order_generated_dataset_list = [l.lower() for l in place_order_generated_dataset_list]
            fallback_generated_dataset_list = [l.lower() for l in fallback_generated_dataset_list]
            menu_cancel_generated_dataset_list = [l.lower() for l in menu_cancel_generated_dataset_list]
            
        # @ 주문 템플릿에서 slot 개수 적은 거 확인
        for line in place_order_generated_dataset_list:
            for slot in cnt_per_slot.keys():
                if slot in line:
                    cnt_per_slot[slot]['cnt'] += line.count(slot)
        # print(f'Before augmentation >> {cnt_per_slot}')
        
        mean_cnt = 0
        for value in cnt_per_slot.values():
            mean_cnt += value['cnt']
            
        # ! 너무 많은 거 같아서 평균의 1/4만 사용
        mean_cnt = mean_cnt // (len(cnt_per_slot.keys()) * 4)

        insufficient_keys = [slot for slot, values in cnt_per_slot.items() if values['cnt'] < mean_cnt]
        print(f'mean_cnt : {mean_cnt}, insufficient_keys : {insufficient_keys}')
        print(f'len: {len(place_order_generated_dataset_list)}')
        
        # @ slot 개수 적은 거 remake -- 메뉴명은 그대로 만드는데 옵션명은 상품명이랑 묶어서 만들기 
        for slot in insufficient_keys:
            cnt = 0    
            make_cnt = mean_cnt - cnt_per_slot[slot]['cnt']
            
            while cnt < make_cnt:
                # 상품명인 경우
                if slot in prod_opt_dict.keys():
                    name = cnt_per_slot[slot]['name']
                    # 메뉴 이름 전처리
                    name_list = prepro_name(name)
                    name = random.choice(name_list)
                    phrase = f'<{name}:{slot}>'
                    place_order_generated_dataset_list = make_templates(place_order_generated_dataset_list, menu_phrase=phrase, option_group_name=None, option_phrase=None, use_option=False)
                    cnt += 1
                # 옵션상품인 경우 
                else: 
                    for menu_code, menu_dict in prod_opt_dict.items():
                        options = menu_dict['options']
                        if options is None:
                            continue
                        if slot in options.keys():
                            option_group_name = list(options.values())[0]['option_group_name']
                            
                            menu_name = menu_dict['menu_name'].replace('<','').replace('>','').replace(':','')
                            menu_name_list = prepro_name(menu_name)
                            menu_name = random.choice(menu_name_list)
                            menu_phrase = f'<{menu_name}:{menu_code}>'
                            
                            option_name = cnt_per_slot[slot]['name']
                            # 메뉴 이름 전처리
                            option_name_list = prepro_name(option_name)
                            option_name = random.choice(option_name_list)
                            option_phrase = f'<{option_name}:{slot}>'
                            
                            place_order_generated_dataset_list = make_templates(place_order_generated_dataset_list, menu_phrase=menu_phrase, option_group_name=None, option_phrase=option_phrase, use_option=True)
                            
                            cnt += 1                
                            
                    # 옵션상품만 
                    option_name = cnt_per_slot[slot]['name']
                    option_name_list = prepro_name(option_name)
                    option_name = random.choice(option_name_list)
                    option_phrase = f'<{option_name}:{slot}>'
                    place_order_generated_dataset_list = make_templates(place_order_generated_dataset_list, menu_phrase=None, option_group_name=None, option_phrase=option_phrase, use_option=False)
                    cnt += 1
                    
                    # 옵션그룹명 추가
                    option_group_name = None
                    place_order_generated_dataset_list = make_templates(place_order_generated_dataset_list, menu_phrase=None, option_group_name=option_group_name, option_phrase=option_phrase, use_option=False)
                    cnt += 1
                    
                            
                    if cnt >= make_cnt :
                        break
                    
        print('finish making insufficient data')
                    
        place_order_generated_dataset_list = [' '.join(line.split()) for line in place_order_generated_dataset_list]
        fallback_generated_dataset_list = [' '.join(line.split()) for line in fallback_generated_dataset_list]
        menu_cancel_generated_dataset_list = [' '.join(line.split()) for line in menu_cancel_generated_dataset_list]
        
        place_order_generated_dataset_list = list(set(place_order_generated_dataset_list))
        fallback_generated_dataset_list = list(set(fallback_generated_dataset_list))
        menu_cancel_generated_dataset_list = list(set(menu_cancel_generated_dataset_list))
        
        place_order_generated_dataset_list = [l.lower() for l in place_order_generated_dataset_list]
        fallback_generated_dataset_list = [l.lower() for l in fallback_generated_dataset_list]
        menu_cancel_generated_dataset_list = [l.lower() for l in menu_cancel_generated_dataset_list]            
        
        
        for line in place_order_generated_dataset_list:
            for slot in cnt_per_slot.keys():
                if slot in line:
                    after_cnt_per_slot[slot]['cnt'] += line.count(slot)
        # print(f'After augmentation >> {after_cnt_per_slot}')
        print(f'len: {len(place_order_generated_dataset_list)}\n')
        
        
        
        intent_data_path = 'utils/intent_data/'
        intent_data_listdir = os.listdir(intent_data_path)
        
        train_dir_path = os.path.join(store_path, 'train', 'intent_data') #, intent_data)
        test_dir_path = os.path.join(store_path, 'test', 'intent_data') #, intent_data)
        
        for intent_data in intent_data_listdir:
            src_path = os.path.join(intent_data_path, intent_data)
            if not os.path.exists(train_dir_path):
                os.makedirs(train_dir_path)
                os.makedirs(test_dir_path)
            
            with open(src_path, 'r', encoding='utf-8-sig') as f:
                raw_lines = f.readlines()
            raw_lines = raw_lines[1:]
            train_ratio = int(len(raw_lines) * 0.9)
            
            train_dst_path = os.path.join(train_dir_path, intent_data)
            test_dst_path = os.path.join(test_dir_path, intent_data)
            
            # TRAIN DATASET
            with open(train_dst_path, 'w', encoding='utf-8-sig') as f:
                for l in raw_lines[:train_ratio]:
                    f.write(l)
                    
            # ? TEST DATASET
            with open(test_dst_path, 'w', encoding='utf-8-sig') as f:
                for l in raw_lines[train_ratio:]:
                    f.write(l)
        
        
        train_place_order_save_path = os.path.join(train_dir_path, 'place_order.csv')
        test_place_order_save_path = os.path.join(test_dir_path, 'place_order.csv')
        train_fallback_save_path = os.path.join(train_dir_path, 'fallback.csv')
        test_fallback_save_path = os.path.join(test_dir_path, 'fallback.csv')
        train_menu_cancel_save_path = os.path.join(train_dir_path, 'menu_cancel.csv')
        test_menu_cancel_save_path = os.path.join(test_dir_path, 'menu_cancel.csv')
        
        with open(train_place_order_save_path, 'w', encoding='utf-8-sig') as f:
            f.write('intent,sentence\n')
            for line in place_order_generated_dataset_list:
                line = line.replace(',', ' ')
                f.write(f'place_order,{line}\n')
        with open(test_place_order_save_path, 'w', encoding='utf-8-sig') as f:
            f.write('intent,sentence\n')
            for line in place_order_generated_test_dataset_list:
                line = line.replace(',', ' ')
                f.write(f'place_order,{line}\n')
                
        fallback_train_ratio = int(len(fallback_generated_dataset_list) * 0.9)
        with open(train_fallback_save_path, 'a', encoding='utf-8-sig') as f:
            for line in fallback_generated_dataset_list[:fallback_train_ratio]:
                line = line.replace(',', ' ')
                f.write(f'fallback,{line}\n')
        with open(train_fallback_save_path, 'a', encoding='utf-8-sig') as f:
            for line in fallback_generated_dataset_list[fallback_train_ratio:]:
                line = line.replace(',', ' ')
                f.write(f'fallback,{line}\n')
                
        cancel_train_ratio = int(len(menu_cancel_generated_dataset_list) * 0.9)
        with open(train_menu_cancel_save_path, 'w', encoding='utf-8-sig') as f:
            f.write('intent,sentence\n')
            for line in menu_cancel_generated_dataset_list[:cancel_train_ratio]:
                line = line.replace(',', ' ')
                f.write(f'menu_cancel,{line}\n')      
        with open(test_menu_cancel_save_path, 'w', encoding='utf-8-sig') as f:
            f.write('intent,sentence\n')
            for line in menu_cancel_generated_dataset_list[cancel_train_ratio:]:
                line = line.replace(',', ' ')
                f.write(f'menu_cancel,{line}\n')      
                    
                    
        # 만든 데이터 합치기
        store_path = os.path.join(data_path, store_no)
        
        for type in ['train', 'test']:
            store_intent_data_dir = os.path.join(os.path.join(store_path, type, 'intent_data'))
            store_intent_data_listdir = os.listdir(store_intent_data_dir)
            store_intent_data_listdir = [intent_data for intent_data in store_intent_data_listdir if intent_data.endswith('csv')]
            if 'all_dataset.csv' in store_intent_data_listdir:
                store_intent_data_listdir.remove('all_dataset.csv')
            
            all_dataset_list = list()
            for intent_data in store_intent_data_listdir:
                intent_data_path = os.path.join(store_intent_data_dir, intent_data)
                if intent_data == 'point.csv':
                    intent_data_list = list()
                    with open(intent_data_path, 'r', encoding='utf-8-sig') as f:
                        reader = csv.reader(f, delimiter='\t')
                        for data in reader:
                            temp = f'{data[0]}\t{data[1]}\n'
                            intent_data_list.append(temp)
                else:
                    with open(intent_data_path, 'r', encoding='utf-8-sig') as f:
                        intent_data_list = f.readlines()
                    intent_data_list = [data.replace(',', '\t') for data in intent_data_list]
                intent_data_list = intent_data_list[1:]
                intent_data_list = list(set(intent_data_list))
                all_dataset_list.extend(intent_data_list)
                
            csv_files_path = os.path.join(store_path, type, 'csv')
            if not os.path.exists(csv_files_path) :
                os.mkdir(csv_files_path)
                
            all_dataset_path = os.path.join(csv_files_path, 'all_dataset.csv')
            with open(all_dataset_path, 'w', encoding='utf-8-sig') as f:
                f.write('intent\tsentence\n')
                for line in all_dataset_list :
                    f.write(line)
                
        # 인텐트 개수 세서 pkl로 저장
        all_intent = [intent.split('\t')[0] for intent in all_dataset_list]
        all_intent = sorted(list(set(all_intent)))
        all_intent_path = os.path.join(store_path, 'intent_list.pkl')
        with open(all_intent_path, 'wb') as f:
            pickle.dump(all_intent, f)

        print(time.ctime())
        due_time = time.time() - start_time
        due_time = str(datetime.timedelta(seconds=due_time)).split('.')[0]
        print(f" *** Time taken:  {due_time}")
    except Exception as e:
        print(f' *** Error raised >>> {e}\n {traceback.format_exc()}')
        raise Exception