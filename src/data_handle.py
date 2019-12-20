#data_handle.py
#(2 (3 (3 Effective) (2 but)) (1 (1 too-tepid) (2 biopic)))
def data_handle(str_temp):
    len_str = len(str_temp)
    result = ""
    score_temp = ""
    while(str_temp[0] == "("):
        end_rank = 0
        while(end_rank < len(str_temp)):
            if(str_temp[end_rank] == ")"):
                break
            end_rank += 1
        start_rank = end_rank
        while start_rank >= 0:
            start_rank -= 1
            if(str_temp[start_rank] == "("):
                break
        kongge_rank = start_rank
        while kongge_rank < end_rank:
            kongge_rank += 1
            if(str_temp[kongge_rank] == " "):
                break
        # print(str_temp[kongge_rank+1:end_rank],str_temp[start_rank+1:kongge_rank])
        score_temp = str_temp[start_rank+1:kongge_rank]
        str_temp = str_temp[:start_rank]+str_temp[kongge_rank+1:end_rank]+str_temp[end_rank+1:]
        
    print(str_temp,score_temp)

input_path = "../data/trees/test.txt"
f_input=open(input_path,'r',encoding = 'utf-8')
lines = f_input.readlines()
for line in lines:
    data_handle(line.strip())