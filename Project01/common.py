# -*- coding: utf-8 -*-
# This file contains the mostly used functions
from collections import defaultdict
import re
import random

# define hamming distance
# Following fuction is copied from python package 'distance'
def hamming(seq1, seq2, normalized=False):
	L = len(seq1)
	if L != len(seq2):
		raise ValueError("expected two strings of the same length")
	if L == 0:
		return 0.0 if normalized else 0  # equal
	dist = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
	if normalized:
		return dist / float(L)
	return dist

# mapping
num_to_str = { 0:'1110111', 1:'0010010', 2:'1011101', 3:'1011011', 4:'0111010', 5:'1101011', 6:'1101111', 7:'1010010', 8:'1111111', 9:'1111011' }
str_to_num = { '0000000':0, '1110111':0, '0010010':1, '0100100':1, '1011101':2, '1011011':3, '0111010':4, '1101011':5, '1101111':6, '0101111':6, '1010010':7, '1111111':8, '1111011':9, '1111010':9 }
oper_encoding = { '-': '000001', '+': '010011', '*': '110101', '=': '101001'}

# construct graph
node_list = [ format(i,'07b') for i in range(128)]
same = defaultdict(list)
minus = defaultdict(list)
plus = defaultdict(list)
for node1 in node_list:
    same[node1] = []
    minus[node1] = []
    plus[node1] = []
    for node2 in node_list:
        if (node1.count('1') == node2.count('1')) and (hamming(node1, node2) == 2):
            same[node1].append(node2)
        elif (node1.count('1') == node2.count('1') + 1) and (hamming(node1, node2) == 1):
            minus[node1].append(node2)
        elif (node1.count('1') == node2.count('1') - 1) and (hamming(node1, node2) == 1):
            plus[node1].append(node2)

#parse equation to corresponding operator and digits
def parse_eq(equation):
    A, B, C = re.split(r'\W',equation)
    operator = re.search(r'\W',equation).group(0)
    A_10 = '0000000' if int(A)//10 == 0 else num_to_str[int(A)//10]
    A_1 = num_to_str[int(A)%10]
    B_10 = '0000000' if int(B)//10 == 0 else num_to_str[int(B)//10]
    B_1 = num_to_str[int(B)%10]
    C_10 = '0000000' if int(C)//10 == 0 else num_to_str[int(C)//10]
    C_1 = num_to_str[int(C)%10]
    
    return operator,[A, B, C],[A_10, A_1, B_10, B_1, C_10, C_1]

#check correct answer 
def check_correct(equation, new_eqs, oper): 
    for candidate in new_eqs:
        if (candidate[0] != '1110111') and (candidate[2] != '1110111') and (candidate[4] != '1110111'):#拒绝十位为0的输入
            new_A_10 = str_to_num[candidate[0]]
            if candidate[1] == '0000000':#个位可以变为空，下同
                if candidate[0] == '0000000':
                    continue
                else:
                    new_A_1 = new_A_10
                    new_A_10 = 0
            else:
                new_A_1 = str_to_num[candidate[1]]          
            new_B_10 = str_to_num[candidate[2]]
            if candidate[3] == '0000000':
                if candidate[2] == '0000000':
                    continue
                else:
                    new_B_1 = new_B_10
                    new_B_10 = 0
            else:
                new_B_1 = str_to_num[candidate[3]]            
            new_C_10 = str_to_num[candidate[4]]
            if candidate[5] == '0000000':
                if candidate[4] == '0000000':
                    continue
                else:
                    new_C_1 = new_C_10
                    new_C_10 = 0           
            else:
                new_C_1 = str_to_num[candidate[5]]  
            new_A = str(10*new_A_10 + new_A_1)
            new_B = str(10*new_B_10 + new_B_1)
            new_C = str(10*new_C_10 + new_C_1)  
            new_exp = new_A + oper + new_B
            if eval(new_exp) == eval(new_C):
                new_equation = new_exp + '=' + new_C
                if (new_equation != equation):
                    return True, candidate # change to candidate after validation
    
    return False, []

# generate new questions
def question_generator():
    oper = random.choice(['-','+','*'])
    if oper == '*':
        a = random.randint(0,11)
        b = random.randint(0,11)
    else:
        a = random.randint(0,99)
        b = random.randint(0,99)

    c = random.randint(0, 99)
    eq_str = str(a) + oper + str(b) + '=' + str(c)

    return eq_str

# find possible solutions
def matching(oper,digits):
    match_cnt = 0
    eq_str = oper_encoding[oper][2:]
    
    for digit in digits:
        match_cnt += digit.count('1') 
        eq_str = eq_str + digit
    match_cnt += (oper_encoding[oper][2:]).count('1') + 2
    oper_det = oper_encoding[oper][:2]
    return match_cnt, eq_str, oper_det

# evaluating the hardship to solve the problem
def hardship(input_eq_str, eq_str, input_oper_det, oper_det):
    dist = hamming(input_eq_str, eq_str)
    oper_dist = hamming(input_oper_det, oper_det)
    if oper_dist == 2:
        if (oper_det[0] == oper_det[1]):
            dist += 1
        else:
            dist -= 1
    hardship = dist // 2
    return hardship