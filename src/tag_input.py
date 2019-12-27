#\((?![^\)]*\(.*?\)).*?\)
#get tag_input
import re
import sys
pattern = re.compile(r'\((?![^\)]*\(.*?\)).*?\)')
def tag_input(str_temp):
    str_temp = str_temp.strip()
    temp_list =  pattern.findall(str_temp )
    temp_list = [(item.split(" ")[0][1:]) for item in temp_list]
    result = " ".join(temp_list)
    result += "\t"+str(str_temp.split(" ")[0][1:])
    return result
input = sys.argv[1]
input_path = "../data/trees/"+input+".txt"
f_input=open(input_path,'r',encoding = 'utf-8')
lines = f_input.readlines()
for line in lines:
    # print(line)
    print(tag_input(line.strip()))