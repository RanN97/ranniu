def create_sequence_label_breadth_first(tree_dict_start_to_category, tree_dict_category_to_first_subclass, tree_dict_first_subclass_to_second_subclass, tree_dict_second_subclass_to_disease, icd_list):

    sequence_list = ['start']
    category_list = []
    first_subclass_list = []
    second_subclass_list = []
    disease_list = []
    for disease in icd_list:
        temp = [disease]
        for k,v in tree_dict_second_subclass_to_disease.items():
            if disease in v:
                second_subclass = k
                temp.append(second_subclass)
                break
        for k,v in tree_dict_first_subclass_to_second_subclass.items():
            if second_subclass in v:
                first_subclass = k
                temp.append(first_subclass)
                break
        for k,v in tree_dict_category_to_first_subclass.items():
            if first_subclass in v:
                category = k
                temp.append(category)
                break
        temp = temp[::-1]
        for i in range(3):
            temp.append('/'+temp[2-i])

        disease_list.append(disease)
        second_subclass_list.append(second_subclass)
        first_subclass_list.append(first_subclass)
        category_list.append(category)

    disease_list = sorted(list(set(disease_list)))
    second_subclass_list = sorted(list(set(second_subclass_list)))
    first_subclass_list = sorted(list(set(first_subclass_list)))
    category_list = sorted(list(set(category_list)))


    sequence_list.extend(category_list)
    for i in category_list[::-1]:
        i_de_first = list(set(first_subclass_list).intersection(tree_dict_category_to_first_subclass[i]))
        i_index = sequence_list.index(i)
        sequence_list.extend(i_de_first)
        for j in i_de_first[::-1]:
            j_de_second = list(set(second_subclass_list).intersection(tree_dict_first_subclass_to_second_subclass[j]))
            j_index = sequence_list.index(j)
            sequence_list.extend(j_de_second)
            for k in j_de_second[::-1]:
                k_de_disease = list(set(disease_list).intersection(tree_dict_second_subclass_to_disease[k]))
                k_index = sequence_list.index(k)
                sequence_list.extend(k_de_disease)
                sequence_list.append('/'+k)
            sequence_list.append('/'+j)
        sequence_list.append('/'+i)
        
    sequence_list.append('/start')

    return sequence_list




def create_sequence_label_depth_first(tree_dict_start_to_category, tree_dict_category_to_first_subclass, tree_dict_first_subclass_to_second_subclass, tree_dict_second_subclass_to_disease, dataframe):

    label_sequencial_list = []
    for i in range(dataframe.shape[0]):
        sequence_list = ['start', '/start']
        for disease in dataframe['ICD9_CODE'].iloc[i]:
            temp = [disease]
            for k,v in tree_dict_second_subclass_to_disease.items():
                if disease in v:
                    second_subclass = k
                    temp.append(second_subclass)
                    break
            for k,v in tree_dict_first_subclass_to_second_subclass.items():
                if second_subclass in v:
                    first_subclass = k
                    temp.append(first_subclass)
                    break
            for k,v in tree_dict_category_to_first_subclass.items():
                if first_subclass in v:
                    category = k
                    temp.append(category)
                    break
            temp = temp[::-1]
            for i in range(3):
                temp.append('/'+temp[2-i])


            if category not in sequence_list:
                insert_index = sequence_list.index('/start')
                sequence_list[insert_index:insert_index] = temp
            else:
                if first_subclass not in sequence_list:
                    insert_index = sequence_list.index('/'+category)
                    sequence_list[insert_index:insert_index] = temp[1:-1]
                else:
                    if second_subclass not in sequence_list:
                        insert_index = sequence_list.index('/'+first_subclass)
                        sequence_list[insert_index:insert_index] = temp[2:-2]
                    else:
                        insert_index = sequence_list.index('/'+second_subclass)
                        sequence_list[insert_index:insert_index] = temp[3:-3]
        
        label_sequencial_list.append(sequence_list[:-1])    # the sequence do not incluede '/start'.
    dataframe['SEQUENCIAL_LABEL'] = label_sequencial_list

    return dataframe