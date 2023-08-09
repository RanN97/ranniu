import re
import pandas as pd
from tqdm import tqdm
import ast
import torch
from torch.utils.data.dataset import Dataset


def text_cleaning(dataframe):
    for i in range(dataframe.shape[0]):
        dataframe['TEXT'].iloc[i] = re.sub("(Admission Date:)|(Discharge Date:)|(Service:)|(Date of Birth:)|(Sex:)|(Attending:)|(Provider:)|(Name:)|(Date/Time:)|(MD Phone:)|(Completed by:)|(Job#:)|(Dictated By:)", "", dataframe['TEXT'].iloc[i]) 
        dataframe['TEXT'].iloc[i] = re.sub(r'\[\*\*.*?\*\*\]', '', dataframe['TEXT'].iloc[i])  # delete [**      **]
        dataframe['TEXT'].iloc[i] = re.sub('[0-9]{2}:[0-9]{2}', '', dataframe['TEXT'].iloc[i])  # delete 12:11
        dataframe['TEXT'].iloc[i] = re.sub('[0-9]{2}:[0-9]{2}AM', '', dataframe['TEXT'].iloc[i])  # delete 12:11AM
        dataframe['TEXT'].iloc[i] = re.sub('[0-9]{2}:[0-9]{2}PM', '', dataframe['TEXT'].iloc[i])  # delete 12:11PM
        dataframe['TEXT'].iloc[i] = re.sub('==+', ' ', dataframe['TEXT'].iloc[i])   # delete redundant space ' '
        dataframe['TEXT'].iloc[i] = re.sub(' +', ' ', dataframe['TEXT'].iloc[i])   # delete redundant space ' '
        dataframe['TEXT'].iloc[i] = re.sub('\n', ' ', dataframe['TEXT'].iloc[i]) # change \n to ' '
        dataframe['TEXT'].iloc[i] = re.sub('# *', ' ', dataframe['TEXT'].iloc[i])
        # dataframe['TEXT'].iloc[i] = re.sub(' - ', ' ', dataframe['TEXT'].iloc[i])
        # dataframe['TEXT'].iloc[i] = re.sub('- ', ' ', dataframe['TEXT'].iloc[i])
        # dataframe['TEXT'].iloc[i] = re.sub(': *', ' ', dataframe['TEXT'].iloc[i])
        # dataframe['TEXT'].iloc[i] = re.sub('[0-9]\. ', ' ', dataframe['TEXT'].iloc[i])
        # dataframe['TEXT'].iloc[i] = re.sub('\* ', ' ', dataframe['TEXT'].iloc[i])
    return dataframe




def convert_icdstr_to_list(df):
    all_icds = []
    for i in list(df['ICD9_CODE']):
        all_icds.append(ast.literal_eval(''.join(i)))
    df['ICD9_CODE'] = all_icds
    return df




def create_classes_descripsions(tree_dict_start_to_category, tree_dict_category_to_first_subclass, tree_dict_first_subclass_to_second_subclass, tree_dict_second_subclass_to_disease):
    
    # create df_category_and_descripsion:
    first_category_txt = open('/u/home/niur/htc_mimic3/data/icd9_classes/Categories.txt')
    dict_category_and_descripsion = {}
    for i in range(20):
        line_splited_list = first_category_txt.readline().split(' ', 1)
        category = line_splited_list[0][1:-1]
        descripsion = line_splited_list[1].strip('\n')
        dict_category_and_descripsion[category] = descripsion
    
    create_df = {'classes':dict_category_and_descripsion.keys(), 'descripsion':dict_category_and_descripsion.values()}
    df_category_and_descripsion = pd.DataFrame(create_df)


    # create df_first_subclassed_and_descripsion:
    first_sub_class_txt = open('/u/home/niur/htc_mimic3/data/icd9_classes/First_sub_classes.txt')
    first_category_txt = open('/u/home/niur/htc_mimic3/data/icd9_classes/Categories.txt')
    dict_first_subclassed_and_descripsion = {}
    first_category_descripsion = first_category_txt.readline().split(' ', 1)[1].strip('\n')
    for i in range(177):
        line_splited_list = first_sub_class_txt.readline().rpartition(' ')
        if line_splited_list[0] == '':
            first_category_descripsion = first_category_txt.readline().split(' ', 1)[-1].strip('\n')
        first_sub_class = line_splited_list[-1][1:-2]
        descripsion = line_splited_list[0].strip('\n')
        dict_first_subclassed_and_descripsion[first_sub_class] = first_category_descripsion + ', ' + descripsion
    del dict_first_subclassed_and_descripsion['']

    create_df = {'classes':dict_first_subclassed_and_descripsion.keys(), 'descripsion':dict_first_subclassed_and_descripsion.values()}
    df_first_subclassed_and_descripsion = pd.DataFrame(create_df)



    # create df_second_subclassed_and_descripsion:
    second_sub_class_txt = open('/u/home/niur/htc_mimic3/data/icd9_classes/Second_sub_classes.txt')
    dict_second_subclassed_and_descripsion = {}
    for i in range(1391):
        line_splited_list = second_sub_class_txt.readline().split(' ',1)
        second_sub_class = line_splited_list[0]
        second_subclass_descripsion = line_splited_list[-1].strip('\n')
        dict_second_subclassed_and_descripsion[second_sub_class] = second_subclass_descripsion
    del dict_second_subclassed_and_descripsion[',\n']
    del dict_second_subclassed_and_descripsion['/\n']

    # dict_first_subclassed_and_descripsion是要加在字符串前面的   '001-009': 'Infectious and parasitic diseases, Intestinal infectious diseases'
    # dict_second_subclassed_and_descripsion是加在字符串后面的    '001': 'Cholera'
    # tree_dict_first_subclass_to_second_subclass 是对应的字典   '001-009': ['001', '002', '003', '004', '005', '006', '007', '008', '009']

    for j in list(tree_dict_first_subclass_to_second_subclass.keys()):   # j '001-009'
        for k in tree_dict_first_subclass_to_second_subclass[j]:
            if k[0] != '/':   # k ['001', '002', '003', '004', '005', '006', '007', '008', '009']
                dict_second_subclassed_and_descripsion[k] = dict_first_subclassed_and_descripsion[j] + ', ' + dict_second_subclassed_and_descripsion[k]
    
    create_df = {'classes':dict_second_subclassed_and_descripsion.keys(), 'descripsion':dict_second_subclassed_and_descripsion.values()}
    df_second_subclassed_and_descripsion = pd.DataFrame(create_df)





    # create df_disease_and_descripsion:
    df_icd_description_long_short = pd.read_csv('/u/home/niur/htc_mimic3/data/ICD9_description_LONG_SHORT_diagnose.csv')
    dict_disease_and_descripsion = {}
    # dict_second_subclassed_and_descripsion  '001': 'Infectious and parasitic diseases, Intestinal infectious diseases, Cholera'
    # tree_dict_second_subclass_to_disease   '001': ['0010', '0011', '0019']
    # dict_disease_and_descripsion  '0010' Cholera due to vibrio cholerae

    for i in range(df_icd_description_long_short.shape[0]):
        dict_disease_and_descripsion[df_icd_description_long_short['DIAGNOSIS CODE'].iloc[i]] = df_icd_description_long_short['LONG DESCRIPTION'].iloc[i]

    for j in list(tree_dict_second_subclass_to_disease.keys()):  # j '001'
        for k in tree_dict_second_subclass_to_disease[j]:
            if k[0] != '/':    # k '0010'
                dict_disease_and_descripsion[k] = dict_second_subclassed_and_descripsion[j] + ', ' + dict_disease_and_descripsion[k]

    create_df = {'classes':list(dict_disease_and_descripsion.keys()), 'descripsion':list(dict_disease_and_descripsion.values())}
    df_disease_and_descripsion = pd.DataFrame(create_df)


    df_classes_and_descripsion = pd.concat([df_category_and_descripsion, df_first_subclassed_and_descripsion, df_second_subclassed_and_descripsion, df_disease_and_descripsion], axis=0, ignore_index=True)

    return df_classes_and_descripsion



def create_class_bert_tokens(df_classes_and_descripsion, tokenizer):

    sentence_list =  list(df_classes_and_descripsion['descripsion'])

    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []    

    for i in range(len(sentence_list)):
        tokenizer_outputs = tokenizer(sentence_list[i], padding='max_length' ,truncation=False, max_length=128, return_tensors='pt')

        input_ids_list.append(tokenizer_outputs['input_ids'])
        # token_type_ids_list.append(tokenizer_outputs['token_type_ids'])
        # attention_mask_list.append(tokenizer_outputs['attention_mask'])

    input_ids_tensor = torch.stack(input_ids_list, dim=0)
    # token_type_idstensor = torch.stack(token_type_ids_list, dim=0)
    # attention_mask_tensor = torch.stack(attention_mask_list, dim=0)

    return input_ids_tensor #, token_type_idstensor, attention_mask_tensor



class Dataset_create_embeddings_from_classdescripsion_for_classsentence_list(Dataset):
    def __init__(self, input_ids_tensor): # , token_type_idstensor, attention_mask_tensor
        self.input_ids_tensor = input_ids_tensor
        # self.token_type_idstensor = token_type_idstensor
        # self.attention_mask_tensor = attention_mask_tensor


    def __len__(self):
        return self.input_ids_tensor.shape[0]

    def __getitem__(self, index):
        return  self.input_ids_tensor[index]# , self.token_type_idstensor[index], self.attention_mask_tensor[index]
    


def create_classes_embeddings(dataloader, clinical_Bert_Model, device):

    with torch.no_grad():
        embeddings_list = []
        for input_ids_tensor in tqdm(dataloader):   # , token_type_ids_tensor, attention_mask_tensor

            input_ids_tensor = input_ids_tensor.to(device)
            # token_type_ids_tensor = token_type_ids_tensor.to(device)
            # attention_mask_tensor = attention_mask_tensor.to(device)

            input_ids_tensor = input_ids_tensor.reshape((1,128))
            # token_type_ids_tensor = token_type_ids_tensor.reshape((1,128))
            # attention_mask_tensor = attention_mask_tensor.reshape((1,128))
            bert_outputs = clinical_Bert_Model(input_ids_tensor)  # , token_type_ids_tensor, attention_mask_tensor

            embeddings_list.append(bert_outputs)
        embeddings_tensor = torch.stack(embeddings_list, dim=0)
        embeddings_tensor = embeddings_tensor.squeeze().detach().to(torch.device('cpu'))
        return embeddings_tensor



def create_class_embeddings_dict(class_lists, class_embeddings_tensor):
    class_embeddings_dict = {}
    for i in range(len(class_lists)):
        class_embeddings_dict[class_lists[i]] = class_embeddings_tensor[i]
    return class_embeddings_dict


def add_end_classes_embeddings(class_embeddings_dict, class_embeddings_start, class_embeddings_end_start, tree_dict_start_to_category, tree_dict_category_to_first_subclass, tree_dict_first_subclass_to_second_subclass, tree_dict_second_subclass_to_disease, device):
    class_embeddings_dict['start'] = class_embeddings_start
    class_embeddings_dict['/start'] = class_embeddings_end_start
    for key in list(tree_dict_category_to_first_subclass.keys()):
        class_embeddings_dict['/'+key] = (class_embeddings_dict[key]+class_embeddings_end_start)/2
    for key in list(tree_dict_first_subclass_to_second_subclass.keys()):
        class_embeddings_dict['/'+key] = (class_embeddings_dict[key]+class_embeddings_end_start)/2
    for key in list(tree_dict_second_subclass_to_disease.keys()):
        class_embeddings_dict['/'+key] = (class_embeddings_dict[key]+class_embeddings_end_start)/2
    class_embeddings_dict['padding'] = torch.tensor([0.0]*768, dtype=torch.float32).to(device, non_blocking=True)
    return class_embeddings_dict




def convert_sequencial_label_to_embedding_tensor(batch_label_index, class_embeddings_dict, index_to_class_label_dict):

    label_embedding_tensor = []
    for single_label_index in batch_label_index:   #(L), (b,L)
        single_embedding_tensor = torch.stack([class_embeddings_dict[index_to_class_label_dict[index.item()]] for index in single_label_index]) # (L,768)
        label_embedding_tensor.append(single_embedding_tensor)

    return  torch.stack(label_embedding_tensor)   # (b,L,768)
    




def create_text_tokens(datafram, tokenizer):

    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []    

    for i in tqdm(range(datafram.shape[0])):

        tokenizer_outputs = tokenizer(datafram['TEXT'].iloc[i], padding='max_length' ,truncation=True, max_length=4*512, return_tensors='pt')
        tokenizer_outputs['input_ids'] = tokenizer_outputs['input_ids'].reshape((4,512))
        tokenizer_outputs['input_ids'][:,0] = 101
        if tokenizer_outputs['input_ids'].shape[0] > 1:
            tokenizer_outputs['input_ids'][:-1,-1] = 102
        # tokenizer_outputs['token_type_ids'] = tokenizer_outputs['token_type_ids'].reshape((4,512))
        # tokenizer_outputs['attention_mask'] = tokenizer_outputs['attention_mask'].reshape((4,512))

        input_ids_list.append(tokenizer_outputs['input_ids'])
        # token_type_ids_list.append(tokenizer_outputs['token_type_ids'])
        # attention_mask_list.append(tokenizer_outputs['attention_mask'])

    input_ids_tensor = torch.stack(input_ids_list, dim=0)
    # token_type_idstensor = torch.stack(token_type_ids_list, dim=0)
    # attention_mask_tensor = torch.stack(attention_mask_list, dim=0)

    return input_ids_tensor #, token_type_idstensor, attention_mask_tensor  # (n,4,512)





class Dataset_create_embeddings_from_tokens_for_text(Dataset):
    def __init__(self, input_ids_tensor):      #, token_type_idstensor, attention_mask_tensor
        self.input_ids_tensor = input_ids_tensor
        # self.token_type_idstensor = token_type_idstensor
        # self.attention_mask_tensor = attention_mask_tensor

    def __len__(self):
        return self.input_ids_tensor.shape[0]

    def __getitem__(self, index):
        return  self.input_ids_tensor[index]     #, self.token_type_idstensor[index], self.attention_mask_tensor[index]


class Clinical_Bert_Model(torch.nn.Module):
    def __init__(self, clinical_DS_bert):
        super(Clinical_Bert_Model, self).__init__()
        self.clinical_DS_bert = clinical_DS_bert
    
    def forward(self, input_ids_tensor):   # , token_type_ids_tensor, attention_mask_tensor

        bert_outputs = self.clinical_DS_bert(input_ids_tensor)['last_hidden_state'][:,0,:]    # , token_type_ids_tensor, attention_mask_tensor

        return bert_outputs
    


def create_text_embeddings(dataloader, clinical_DS_bert_model, device):
    with torch.no_grad():
        embeddings_list = []
        for input_ids_tensor in tqdm(dataloader): # , token_type_ids_tensor, attention_mask_tensor

            input_ids_tensor = input_ids_tensor.to(device, non_blocking=True)
            # token_type_ids_tensor = token_type_ids_tensor.to(device)
            # attention_mask_tensor = attention_mask_tensor.to(device)

            input_ids_tensor = input_ids_tensor.reshape((4,512))
            # token_type_ids_tensor = token_type_ids_tensor.reshape((4,512))
            # attention_mask_tensor = attention_mask_tensor.reshape((4,512))
            bert_outputs = clinical_DS_bert_model(input_ids_tensor).last_hidden_state[:,0,:]  # , token_type_ids_tensor, attention_mask_tensor
            embeddings_list.append(bert_outputs)
        text_embeddings_tensor = torch.stack(embeddings_list, dim=0)
        text_embeddings_tensor = text_embeddings_tensor.detach().to(torch.device('cpu'))

        return text_embeddings_tensor   # return 2-dim tensor  (n,4,768)
def create_classes_embeddings(dataloader, clinical_DS_bert_model, device):

    with torch.no_grad():
        embeddings_list = []
        for input_ids_tensor in tqdm(dataloader):   # , token_type_ids_tensor, attention_mask_tensor

            input_ids_tensor = input_ids_tensor.to(device, non_blocking=True)
            # token_type_ids_tensor = token_type_ids_tensor.to(device)
            # attention_mask_tensor = attention_mask_tensor.to(device)

            input_ids_tensor = input_ids_tensor.reshape((1,128))
            # token_type_ids_tensor = token_type_ids_tensor.reshape((1,128))
            # attention_mask_tensor = attention_mask_tensor.reshape((1,128))
            bert_outputs = clinical_DS_bert_model(input_ids_tensor)   # , token_type_ids_tensor, attention_mask_tensor

            embeddings_list.append(bert_outputs)
            embeddings_tensor = torch.stack(embeddings_list, dim=0)
            embeddings_tensor = embeddings_tensor.squeeze().detach()
        return embeddings_tensor




def create_tokens_mask_dict(tree_dict_start_to_category, tree_dict_category_to_first_subclass, tree_dict_first_subclass_to_second_subclass, tree_dict_second_subclass_to_disease):

    all_tokens_list = ['start']                        # start

    all_tokens_list.extend(list(tree_dict_category_to_first_subclass.keys()))  # category

    all_tokens_list.append('/start')                                # /start

    all_tokens_list.extend(list(tree_dict_first_subclass_to_second_subclass.keys()))   # first_subclass

    all_tokens_list.extend(list(tree_dict_second_subclass_to_disease.keys()) )     # second_subclass

    for k,v in tree_dict_second_subclass_to_disease.items():    # disease
        for key in v:
            if key[0] != '/':
                all_tokens_list.append(key)

    for key in list(tree_dict_category_to_first_subclass.keys()):   # /category
        all_tokens_list.append('/'+key)

    for key in list(tree_dict_first_subclass_to_second_subclass.keys()):   # /first_subclass
        all_tokens_list.append('/'+key)

    for key in list(tree_dict_second_subclass_to_disease.keys()):   # /second_subclass
        all_tokens_list.append('/'+key)
    all_tokens_list.append('padding')







    mask_dict_tokens = {}
    for key in list(tree_dict_start_to_category.keys()):       # start's following
        temp_index_list = []
        temp_tensor = torch.tensor([(torch.inf)*(-1)]*17391)
        for token in tree_dict_start_to_category[key]:
            if token[0] != '/':
                temp_index_list.append(all_tokens_list.index(token))
        temp_tensor[temp_index_list] = 0
        mask_dict_tokens[key] = temp_tensor

    for key in list(tree_dict_category_to_first_subclass.keys()):      # category's following
        temp_index_list = []
        temp_tensor = torch.tensor([(torch.inf)*(-1)]*17391)
        for token in tree_dict_category_to_first_subclass[key]:
            if token[0] != '/':
                temp_index_list.append(all_tokens_list.index(token))
        temp_tensor[temp_index_list] = 0
        mask_dict_tokens[key] = temp_tensor


    mask_dict_tokens['/start'] = torch.tensor([(torch.inf)*(-1)]*17391)  # /start's following



    for key in list(tree_dict_first_subclass_to_second_subclass.keys()):   # first_subclass's following
        temp_index_list = []
        temp_tensor = torch.tensor([(torch.inf)*(-1)]*17391)
        for token in tree_dict_first_subclass_to_second_subclass[key]:
            if token[0] != '/':
                temp_index_list.append(all_tokens_list.index(token))
        temp_tensor[temp_index_list] = 0
        mask_dict_tokens[key] = temp_tensor

    for key in list(tree_dict_second_subclass_to_disease.keys()):   # second_subclass's following
        temp_index_list = []
        temp_tensor = torch.tensor([(torch.inf)*(-1)]*17391)
        for token in tree_dict_second_subclass_to_disease[key]:
            if token[0] != '/':
                temp_index_list.append(all_tokens_list.index(token))
        temp_tensor[temp_index_list] = 0
        mask_dict_tokens[key] = temp_tensor


    for k,v in tree_dict_second_subclass_to_disease.items():   # disease's following
        temp_index_list = []
        temp_tensor = torch.tensor([(torch.inf)*(-1)]*17391)
        for token in v:
            temp_index_list.append(all_tokens_list.index(token))
        # temp_index_list.append(all_tokens_list.index('/'+k))
        temp_tensor[temp_index_list] = 0
        for disease in v:
            if disease[0] != '/':
                mask_dict_tokens[disease] = temp_tensor


    for key in list(tree_dict_category_to_first_subclass.keys()):   # /category's following
        temp_index_list = []
        temp_tensor = torch.tensor([(torch.inf)*(-1)]*17391)
        for token in list(tree_dict_category_to_first_subclass.keys()):
            temp_index_list.append(all_tokens_list.index(token))     # can add if to avoid predicting repeatly
        temp_index_list.append(all_tokens_list.index('/start'))
        temp_tensor[temp_index_list] = 0
        mask_dict_tokens['/'+key] = temp_tensor

    for key in list(tree_dict_first_subclass_to_second_subclass.keys()):   # /first_subclass's following
        temp_index_list = []
        temp_tensor = torch.tensor([(torch.inf)*(-1)]*17391)
        for k,v in tree_dict_category_to_first_subclass.items():
            if key in v:
                for token in tree_dict_category_to_first_subclass[k]:
                    temp_index_list.append(all_tokens_list.index(token))
                temp_index_list.append(all_tokens_list.index('/'+k))
                break
        temp_tensor[temp_index_list] = 0
        mask_dict_tokens['/'+key] = temp_tensor

    for key in list(tree_dict_second_subclass_to_disease.keys()):   # /second_subclass's following
        temp_index_list = []
        temp_tensor = torch.tensor([(torch.inf)*(-1)]*17391)
        for k,v in tree_dict_first_subclass_to_second_subclass.items():
            if key in v:
                for token in tree_dict_first_subclass_to_second_subclass[k]:
                    temp_index_list.append(all_tokens_list.index(token))
                temp_index_list.append(all_tokens_list.index('/'+k))
                break
        temp_tensor[temp_index_list] = 0
        mask_dict_tokens['/'+key] = temp_tensor
    mask_dict_tokens['padding'] = torch.tensor([0]*17391)

    return mask_dict_tokens


def create_class_label_index_dict(mask_dict):
    class_label_index_dict = {}
    c = int(0)
    for item in list(mask_dict.keys()):
        class_label_index_dict[item] = c
        c += 1
    return class_label_index_dict


def create_index_to_class_label_dict(mask_dict):
    index_to_class_label_dict ={}
    class_list = list(mask_dict.keys())
    for i in range(len(class_list)):
        index_to_class_label_dict[int(i)] = class_list[i]
    return index_to_class_label_dict


def convrt_sequencial_label_to_label_index(class_label_index_dict, sequencial_label_list):
    label_index_list = []
    for sequence in sequencial_label_list:
        temp_list = []
        for token in sequence:
            temp_list.append(class_label_index_dict[token])
        label_index_list.append(torch.tensor(temp_list, dtype=torch.long))
    return label_index_list




def split_predicted_class_tokens(predicted_token_list, tree_dict_start_to_category, tree_dict_category_to_first_subclass, tree_dict_first_subclass_to_second_subclass, tree_dict_second_subclass_to_disease):
    predicted_category_tokens_list = []
    predicted_first_class_tokens_list = []
    predicted_second_class_tokens_list = []
    predicted_disease_tokens_list = []
    for token in predicted_token_list:
        if token in list(tree_dict_category_to_first_subclass.keys()):
            predicted_category_tokens_list.append(token)
        elif token in list(tree_dict_first_subclass_to_second_subclass.keys()):
            predicted_first_class_tokens_list.append(token)
        elif token in list(tree_dict_second_subclass_to_disease.keys()):
            predicted_second_class_tokens_list.append(token)
        elif token[0] != '/':
            predicted_disease_tokens_list.append(token)
    return predicted_category_tokens_list, predicted_first_class_tokens_list, predicted_second_class_tokens_list, predicted_disease_tokens_list



def create_0_1_tensor_for_f1(single_sequencial_label, predicted_disease_tokens_list):
    disease_list = list(pd.read_csv('/u/home/niur/htc_mimic3/data/ICD9_description_LONG_SHORT_diagnose.csv')['DIAGNOSIS CODE'])
    label_disease_tokens = list(set(disease_list).intersection(set(single_sequencial_label)))
    label_f1_0_1_tensor = torch.tensor([0]*14567, dtype=int)
    predicted_f1_0_1_tensor = torch.tensor([0]*14567, dtype=int)
    for i in label_disease_tokens:
        index = disease_list.index(i)
        label_f1_0_1_tensor[index] = 1
    for j in predicted_disease_tokens_list:
        index = disease_list.index(j)
        predicted_f1_0_1_tensor[index] = 1
    return label_f1_0_1_tensor, predicted_f1_0_1_tensor
    

def create_probability_tensor_for_AUROC(single_sequencial_label, class_label_index_dict, predicted_token_probabiluty_tensor, predicted_index_list, device):
    class_list = list(class_label_index_dict.keys())
    logits_max = torch.max(predicted_token_probabiluty_tensor, dim=-1)
    probability_AUROC_predicted = torch.tensor([0.0]*17391, dtype=torch.float).to(device, non_blocking=True)

    probability_AUROC_predicted[predicted_index_list] = logits_max[0].type(torch.float)

    label_AUROC_0_1_tensor = torch.tensor([0]*17391)
    for i in single_sequencial_label:
        index = class_list.index(i)
        label_AUROC_0_1_tensor[index] = 1
    

    return probability_AUROC_predicted, label_AUROC_0_1_tensor  # (C), (C)


def create_lable_for_loss_tensor(batch_label_index, end_start_index_label, device):
    label_for_loss_list = []
    for i in range(batch_label_index.shape[0]):
        index_list = batch_label_index[i].tolist()
        end_token_index = index_list.index(17391) if 17391 in index_list else len(index_list)
        index_list.insert(end_token_index, end_start_index_label)
        label_for_loss_list.append(torch.tensor(index_list[1:]))
    lable_for_loss_tensor = torch.stack(label_for_loss_list).to(device, non_blocking=True)

    return lable_for_loss_tensor   # (b,L)