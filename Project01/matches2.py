# -*- coding: utf-8 -*-
# This file contains functions to search for answers of two-matches change
import common as cm
import matches1

# check if correct answer can be achieved by involving changes of operator or equal sign
def move_oper_2(equation):
	oper,nums,digits = cm.parse_eq(equation)
	new_eqs = []
	ans = digits[:]
	
	# 1. 移位问题
	if (digits[0] == '0000000') and (digits[2] == '0000000') and (digits[4] != '0000000'):
		if (oper == '+') or (oper == '*'):
			ans = ['0000000', digits[4], '0000000', digits[5], digits[1], digits[3]]
			new_eqs.append(list(ans))
		else:#移动‘=’
			ans = ['0000000', digits[1], digits[3], digits[4], '0000000', digits[5]]
			new_eqs.append(list(ans))
	elif (digits[0] == '0000000') and (digits[2] != '0000000'):
		if (oper == '=') and (digits[4] == '0000000'):#移动‘=’
			ans = ['0000000', digits[1], '0000000', digits[2], digits[3], digits[5]]
			new_eqs.append(list(ans))
		else:
			ans = [ digits[1], digits[2], '0000000', digits[3], digits[4], digits[5]]
			new_eqs.append(list(ans))
	elif (digits[0] != '0000000') and (digits[2] == '0000000') and (digits[4] != '0000000'):
		if (oper == '+') or (oper == '*'):
			ans = ['0000000', digits[0], digits[1], digits[3], digits[4], digits[5]]
			new_eqs.append(list(ans))
		else:#移动‘=’
			ans = [ digits[0], digits[1], digits[3], digits[4], '0000000', digits[5]]
			new_eqs.append(list(ans))
	elif (oper == '=') and (digits[0] != '0000000') and (digits[2] != '0000000') and (digits[4] == '0000000'):
		ans = [ digits[0], digits[1], '0000000', digits[2], digits[3], digits[5]]
		new_eqs.append(list(ans))
	
	hasfound, solution = cm.check_correct(equation,new_eqs,oper)
	if (hasfound == True):
		if (digits[0] == '0000000') and (digits[2] == '0000000') and (digits[4] != '0000000'):
			if (oper == '+') or (oper == '*'):
				final_solution = [solution[4], solution[5], solution[0], solution[1], solution[2], solution[3]]
				return True, final_solution, oper , True
		else:
			final_solution = solution
			return True, final_solution, oper , False

	#2. 变符号问题
	#2.1
	if (oper == '-'):
		# 2.1.1 ’-‘ 变成 ‘+’: minus1 + search1
		for i in range(6):
			new_eqs = []
			ans = digits[:]
			solution = []
			candidates = cm.minus.get(ans[i])
			for new_digit in candidates:
				ans[i] = new_digit
				new_eqs.append(list(ans))	
			for new_digits in new_eqs:
				hasfound, solution, __, __ = matches1.move_digit_1(equation,'+',new_digits)
				if (hasfound == True):
					final_solution = solution
					return True, final_solution, '+' , False

		# 2.1.2 ’-‘ 变成 ‘*’: minus1		
		for i in range(6):
			new_eqs = []
			ans = digits[:]
			solution = []
			candidates = cm.minus.get(ans[i])
			for new_digit in candidates:
				ans[i] = new_digit
				if set(ans).issubset(set(cm.str_to_num.keys())):
					new_eqs.append(list(ans))
			hasfound, solution = cm.check_correct(equation, new_eqs, '*')
			if (hasfound == True):
				final_solution = solution
				return True, final_solution, '*' , False
		# 2.1.3 ’-‘ 变成 ‘=’:
		# 2.1.3.1 互换 + search_digit_1
		solution = []
		new_digits = [digits[2], digits[3], digits[4], digits[5], digits[0], digits[1]]
		hasfound, solution, __, __ = matches1.move_digit_1(equation,'-',new_digits)
		if (hasfound == True):
			final_solution =  [solution[4], solution[5], solution[0], solution[1], solution[2], solution[3]]
			return True, final_solution, '-' , True
		# 2.1.3.2 minus1, check A=B+C
		for i in range(6):
			ans = digits[:]
			new_eqs = []
			solution = []
			candidates = cm.minus.get(ans[i])
			for new_digit in candidates:
				ans[i] = new_digit
				if set(ans).issubset(set(cm.str_to_num.keys())):
					new_ans = [ans[2], ans[3], ans[4], ans[5], ans[0], ans[1]]
					new_eqs.append(list(new_ans))
			hasfound, solution = cm.check_correct(equation, new_eqs, '+')
			if (hasfound == True):
				final_solution =  [solution[4], solution[5], solution[0], solution[1], solution[2], solution[3]]
				return True, final_solution, '+' , True
		# 2.1.3.3 minus1(-to=) + plus1(=to-)
		for i in range(6):
			ans = digits[:]
			new_eqs = []
			ans_stack = []
			solution = []
			candidates_1 = cm.minus.get(ans[i])
			for new_digit_1 in candidates_1:
				ans[i] = new_digit_1
				ans_stack.append(list(ans))
				for j in range(6):
					candidates_2 = cm.plus.get(ans[j])
					for new_digit_2 in candidates_2:
						ans[j] = new_digit_2
						if set(ans).issubset(set(cm.str_to_num.keys())):
							new_eqs.append(list(ans))
					ans = (ans_stack[-1])[:]
				ans_stack.pop()	
			hasfound, solution = cm.check_correct(equation, new_eqs, '-')
			if (hasfound == True):
				final_solution =  [solution[4], solution[5], solution[0], solution[1], solution[2], solution[3]]
				return True, final_solution, '-' , True
		# 2.1.3.4 (=to-)plus1-(-to=)minus1
		for i in range(6):
			ans = digits[:]
			new_eqs = []
			ans_stack = []
			solution = []
			candidates_1 = cm.plus.get(ans[i])
			for new_digit_1 in candidates_1:
				ans[i] = new_digit_1
				ans_stack.append(list(ans))
				for j in range(6):
					candidates_2 = cm.minus.get(ans[j])
					for new_digit_2 in candidates_2:
						ans[j] = new_digit_2
						if set(ans).issubset(set(cm.str_to_num.keys())):
							new_eqs.append(list(ans))
					ans = (ans_stack[-1])[:]
				ans_stack.pop()
			hasfound, solution = cm.check_correct(equation, new_eqs, '-')
			if (hasfound == True):
				final_solution =  [solution[4], solution[5], solution[0], solution[1], solution[2], solution[3]]
				return True, final_solution, '-' , True

	#2.2
	if (oper == '+'):
		# 2.2.1 ’+‘ 变成 ‘-’: plus1 + search1
		for i in range(6):
			ans = digits[:]
			new_eqs = []
			solution = []
			candidates = cm.plus.get(ans[i])
			for new_digit in candidates:
				ans[i] = new_digit
				new_eqs.append(list(ans))
			for new_digits in new_eqs:
				hasfound, solution, __, __ = matches1.move_digit_1(equation,'-',new_digits)
				if (hasfound == True):
					final_solution = solution
					return True, final_solution, '-' , False
		# 2.2.2 ’+‘ 变成 ‘*’:
		if (int(nums[0]) * int(nums[1]) == int(nums[2])):
			final_solution = digits[:]
			return True, final_solution, '*' , False
		# 2.2.3 ’+‘ 变成 ‘=’
		# 2.2.3.1 互换
		if (int(nums[1]) + int(nums[2]) == int(nums[0])):
			final_solution = digits[:]
			return True, final_solution, '+' , True
		# 2.2.3.2 plus1, check A=B-C
		for i in range(6):
			ans = digits[:]
			new_eqs = []
			solution = []
			candidates = cm.plus.get(ans[i])
			for new_digit in candidates:
				ans[i] = new_digit
				if set(ans).issubset(set(cm.str_to_num.keys())):
					new_ans = [ans[2], ans[3], ans[4], ans[5], ans[0], ans[1]]
					new_eqs.append(list(new_ans))
			hasfound, solution = cm.check_correct(equation, new_eqs, '-')
			if (hasfound == True):
				final_solution =  [solution[4], solution[5], solution[0], solution[1], solution[2], solution[3]]
				return True, final_solution, '-' , True

	#2.3
	if (oper == '*'):
		# 2.3.1 ’*‘ 变成 ‘-’:
		for i in range(6):
			ans = digits[:]
			new_eqs = []
			ans_stack = []
			solution = []
			candidates = cm.plus.get(ans[i])
			for new_digit in candidates:
				ans[i] = new_digit
				if set(ans).issubset(set(cm.str_to_num.keys())):
					new_eqs.append(list(ans))	
			hasfound, solution = cm.check_correct(equation, new_eqs, '-')
			if (hasfound == True):
				final_solution = solution
				return True, final_solution, '-' , False
			
		# 2.3.2 ’*‘ 变成 ‘+’:
		if (int(nums[0]) + int(nums[1]) == int(nums[2])):
			final_solution = digits[:]
			return True, final_solution, '+' , False
	
	return False, [], '', False

# check if correct answer can be achieved only by changes of digits
def move_digit_2(equation):
	oper,__,digits = cm.parse_eq(equation)
	
	for i in range(6):  
		ans = digits[:]
		ans_stack = []
		new_eqs = []
		solution = []

		candidates_1 = cm.same.get(ans[i])# there's a change within same digit
		for new_digit_1 in candidates_1:
			ans[i] = new_digit_1
			ans_stack.append(list(ans))
			for j in range(i,6):
				candidates_2 = cm.same.get(ans[j])
				for new_digit_2 in candidates_2:
					ans[j] = new_digit_2
					if set(ans).issubset(set(cm.str_to_num.keys())):
						new_eqs.append(list(ans))#another change occurs within same digit again (same1-same1)
				ans = (ans_stack[-1])[:]

				candidates_2 = cm.minus.get(ans[j])
				for new_digit_2 in candidates_2:
					ans[j] = new_digit_2
					ans_stack.append(list(ans))
					for k in range(j+1,6):
						candidates_3 = cm.plus.get(ans[k])#another change is removing from former digit and add it to the latter (same1-minus1-plus1)              
						for new_digit_3 in candidates_3:
							ans[k] = new_digit_3
							if set(ans).issubset(set(cm.str_to_num.keys())):
								new_eqs.append(list(ans))
						ans = (ans_stack[-1])[:]
					ans_stack.pop()
				ans = (ans_stack[-1])[:]

				candidates_2 = cm.plus.get(ans[j])
				for new_digit_2 in candidates_2:
					ans[j] = new_digit_2
					ans_stack.append(list(ans))
					for k in range(j+1,6):
						candidates_3 = cm.minus.get(ans[k])#another change is removing from latter digit and add it to the former (same1-plus1-minus1)
						for new_digit_3 in candidates_3:
							ans[k] = new_digit_3
							if set(ans).issubset(set(cm.str_to_num.keys())): 
								new_eqs.append(list(ans))
						ans = (ans_stack[-1])[:]
					ans_stack.pop()
				ans = (ans_stack[-1])[:]
			ans_stack.pop()
		hasfound, solution = cm.check_correct(equation, new_eqs, oper)
		if (hasfound == True):
			final_solution = solution
			return True, final_solution, oper , False

		ans = digits[:]
		ans_stack = []
		new_eqs = []
		solution = []

		candidates_1 = cm.minus.get(ans[i])# remove a match from the first digit
		for new_digit_1 in candidates_1:
			ans[i] = new_digit_1
			ans_stack.append(list(ans))
			for j in range(i+1,6):
				candidates_2 = cm.plus.get(ans[j])# another following digit gets the match
				for new_digit_2 in candidates_2:
					ans[j] = new_digit_2
					ans_stack.append(list(ans))
					for k in range(i,6):
						candidates_3 = cm.same.get(ans[k])# another change occurs within the same digit (minus1-plus1-same1)               
						for new_digit_3 in candidates_3:
							ans[k] = new_digit_3
							if set(ans).issubset(set(cm.str_to_num.keys())):
								new_eqs.append(list(ans))
						ans = (ans_stack[-1])[:]

						candidates_3 = cm.minus.get(ans[k])
						for new_digit_3 in candidates_3:
							ans[k] = new_digit_3
							ans_stack.append(list(ans))
							for m in range(k+1,6):
								candidates_4 = cm.plus.get(ans[m])# another change is removing from former digit and add it to the latter (minus1-plus1-minus1-plus1)
								for new_digit_4 in candidates_4:
									ans[m] = new_digit_4
									if set(ans).issubset(set(cm.str_to_num.keys())):
										new_eqs.append(list(ans))
								ans = (ans_stack[-1])[:]
							ans_stack.pop()
						ans = (ans_stack[-1])[:]

						candidates_3 = cm.plus.get(ans[k])
						for new_digit_3 in candidates_3:
							ans[k] = new_digit_3
							ans_stack.append(list(ans))
							for m in range(k+1,6):
								candidates_4 = cm.minus.get(ans[m])#another change is removing from latter digit and add it to the former (minus1-plus1-plus1-minus1)
								for new_digit_4 in candidates_4:
									ans[m] = new_digit_4
									if set(ans).issubset(set(cm.str_to_num.keys())):
										new_eqs.append(list(ans))
								ans = (ans_stack[-1])[:]
							ans_stack.pop()
						ans = (ans_stack[-1])[:]
					ans_stack.pop()
				ans = (ans_stack[-1])[:]
			ans_stack.pop()

		hasfound, solution = cm.check_correct(equation, new_eqs, oper)
		if (hasfound == True):
			final_solution = solution
			return True, final_solution, oper , False
		
		ans = digits[:]
		ans_stack = []
		new_eqs = []
		solution = []

		candidates_1 = cm.plus.get(ans[i])# add a match to the first digit
		for new_digit_1 in candidates_1:
			ans[i] = new_digit_1
			ans_stack.append(list(ans))
			for j in range(i+1,6):
				candidates_2 = cm.minus.get(ans[j])# another following digit loses the match
				for new_digit_2 in candidates_2:
					ans[j] = new_digit_2
					ans_stack.append(list(ans))
					for k in range(i,6):
						candidates_3 = cm.same.get(ans[k])# another change occurs within the same digit (plus1-minus1-same1)           
						for new_digit_3 in candidates_3:
							ans[k] = new_digit_3
							if set(ans).issubset(set(cm.str_to_num.keys())):
								new_eqs.append(list(ans))
						ans = (ans_stack[-1])[:]

						candidates_3 = cm.minus.get(ans[k])
						for new_digit_3 in candidates_3:
							ans[k] = new_digit_3
							ans_stack.append(list(ans))
							for m in range(k+1,6):
								candidates_4 = cm.plus.get(ans[m])#another change is removing from former digit and add it to the latter (plus1-minus-minus1-plus1)
								for new_digit_4 in candidates_4:
									ans[m] = new_digit_4
									if set(ans).issubset(set(cm.str_to_num.keys())):
										new_eqs.append(list(ans))
								ans = (ans_stack[-1])[:]
							ans_stack.pop()
						ans = (ans_stack[-1])[:]

						candidates_3 = cm.plus.get(ans[k])
						for new_digit_3 in candidates_3:
							ans[k] = new_digit_3
							ans_stack.append(list(ans))
							for m in range(k+1,6):
								candidates_4 = cm.minus.get(ans[m])#another change is removing from latter digit and add it to the former (plus1-minus1-plus1-minus1)
								for new_digit_4 in candidates_4:
									ans[m] = new_digit_4
									if set(ans).issubset(set(cm.str_to_num.keys())):
										new_eqs.append(list(ans))
								ans = (ans_stack[-1])[:]
							ans_stack.pop()
						ans = (ans_stack[-1])[:]
					ans_stack.pop()
				ans = (ans_stack[-1])[:]
			ans_stack.pop()

		hasfound, solution = cm.check_correct(equation, new_eqs, oper)
		if (hasfound == True):
			final_solution = solution
			return True, final_solution, oper , False
	
	return False, [], '', False