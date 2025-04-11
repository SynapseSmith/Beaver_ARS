def divide_prod_n_opt(self):
    if self.tmp_prod_opt_dict['only_prod'] ==[] and self.tmp_prod_opt_dict['both_prod_opt'] ==[]:
        # 아무것도 뽑히지 않은 경우 or 옵션상품이 뽑힌 경우
        return
    
    # 상품O & 옵션상품X 인 메뉴
        # 상품이 하나 이상인 경우 -- # ! 11/28 수정 -- 앞에 거만 뽑기
    if len(self.tmp_prod_opt_dict['only_prod']) > 1:
        self.prod = self.tmp_prod_opt_dict['only_prod'][0]
        
    elif len(self.tmp_prod_opt_dict['only_prod']) == 1:
        # 상품이 하나인 경우
        self.prod = self.tmp_prod_opt_dict['only_prod'][0]
        
    else:
        # 상품이 없는 경우
        pass
        
    # 상품O & 옵션상품O 인 메뉴
        # 옵션상품이 하나인 경우
    if len(self.tmp_prod_opt_dict['both_prod_opt']) == 1:
        if self.prod is None:
            # 상품 되기
            self.prod = self.tmp_prod_opt_dict['both_prod_opt'][0]
        else:
            # 옵션상품 되기
            self.opt.append(self.tmp_prod_opt_dict['both_prod_opt'][0])
    
        # 옵션상품이 하나 이상인 경우
    elif len(self.tmp_prod_opt_dict['both_prod_opt']) > 1:
        for slot_cd in self.tmp_prod_opt_dict['both_prod_opt']:
            if self.prod is None:
                # 상품 되기
                self.prod = slot_cd
            else:    
                # 옵션상품 되기
                if slot_cd not in self.opt:
                    self.opt.append(slot_cd)
    else:
        self.logger.info('\t\tNO OPTIONS')

                
def function_by_slots(self, pred_slot, len_tokens, tokeized_words_list, uncertain_idx_list):
    """
    idx
    0 ~ 1 : 숫자
    마지막 idx : O
    """
    # self.logger.info(f'\tpred_slot >>> {pred_slot}')
    # self.temp_slot_text = ''

    
    for idx in range(len_tokens):
        if idx in uncertain_idx_list:
            continue
        # 확실한 슬롯만
        else:
            slot = pred_slot[idx]
            
            # 숫자
            if slot in [0, 1] :
                self.tmp_num = tokeized_words_list[idx]
                
                # @ phone_number나 point 인텐트 - num을 str 형태로 합쳐야함
                if self.intent == 'phone_number' :
                    self.phone_num += str(self.convert_str_to_int(self.tmp_num))
                # ! 잘 분류하는 지 봐야함 --> 그렇지 않다면 추가 로직 필요(이만오천 -> 자릿수 구분..)
                elif self.intent == 'point':
                    self.point += str(self.convert_str_to_int(self.tmp_num))
                
                # @ 일반 경우 - 수량, 번호, 할부개월 선택
                else:
                    if isinstance(self.num, int) and self.tmp_num is not None:
                        self.num += self.convert_str_to_int(self.tmp_num)
                    elif self.tmp_num is not None:
                        self.num = self.convert_str_to_int(self.tmp_num)
            
            # 그 외 상품명/옵션상품명 슬롯
            else:
                slot_cd = self.store['bi_slot_list'][slot].split('-')[-1]
                
                if slot_cd != 'O':    
                    """11/20
                        self.tmp_prod_opt_dict에 분류해서 담기
                        후에 self.divide_prod_n_opt로 분류
                        
                        only_prod --> 상품O & 옵션상품X
                        both_prod_opt --> 상품O & 옵션상품O
                    """
                    # 상품O & 옵션상품X
                    if slot_cd in self.store['prod_opt_dict'].keys() and slot_cd not in self.store['opt_cd_dict'].keys():
                        if slot_cd not in self.tmp_prod_opt_dict['only_prod']:
                            self.tmp_prod_opt_dict['only_prod'].append(slot_cd)
                        
                    # 상품O & 옵션상품O
                    if slot_cd in self.store['prod_opt_dict'].keys() and slot_cd in self.store['opt_cd_dict'].keys():
                        if slot_cd not in self.tmp_prod_opt_dict['both_prod_opt']:
                            self.tmp_prod_opt_dict['both_prod_opt'].append(slot_cd)
                    
                    # 상품X & 옵션상품O
                    if slot_cd not in self.opt and slot_cd not in self.store['prod_opt_dict'].keys():
                        self.opt.append(slot_cd)
                        
                    # if self.temp_slot_text != '':
                    #     self.temp_slot_text += ' '
                    # self.temp_slot_text += tokeized_words_list[idx]

    
    if self.intent == 'select_number':
        if self.num is None:
            self.logger.info(f' Intent is {self.intent}, but NUM slot is None')
            self.set_replay_format()
        else:
            self.output['body']['respNo'] = self.num            
        return
    elif self.intent == 'select_quantity':
        if self.num is None:
            self.logger.info(f' Intent is {self.intent}, but NUM slot is None')
            self.set_replay_format()
        else:
            self.output['body']['qty'] = self.num
        return
    elif self.intent == 'point':
        if self.point is None:
            self.logger.info(f' Intent is {self.intent}, but POINT slot is None')
            self.set_replay_format()
        else:
            self.output['body']['point'] = int(self.point)
        return
    elif self.intent == 'phone_number':
        self.phone_num = self.phone_num[:11]
        if self.phone_num is None:
            self.logger.info(f' Intent is {self.intent}, but PHONE_NUM slot is None')
            self.set_replay_format()
        else:
            self.output['body']['phoneNumber'] = str(self.phone_num)
        return
    elif self.intent == 'installment_pay':
        if self.num is None:
            self.logger.info(f' Intent is {self.intent}, but NUM slot is None')
            self.set_replay_format()
        else:
            self.output['body']['payMonth'] = int(self.num)
        return

    elif self.intent == 'place_order' : # place_order
        # 상품/옵션상품명 분류
        self.divide_prod_n_opt()

        if self.num is not None:
            self.output['body']['qty'] = self.num