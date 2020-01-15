# -*- coding: utf-8 -*-
# This file contains functions to search for answers of one-match change
import common as cm

# check if correct answer can be achieved by moving a match of the operator
def move_oper_1(equation):
	oper,nums,digits = cm.parse_eq(equation)
	#change - to +, so a match should be removed from digits (minus1) 
	if (oper == '-'):
		if (int(nums[1])-int(nums[2]) == int(nums[0])):
			final_solution = digits[:]
			return True, final_solution, '-' , True
		else:
			for i in range(6):
				new_eqs = []
				ans = digits[:]
				candidates = cm.minus.get(digits[i])
				for new_digit in candidates:
					if (new_digit in cm.str_to_num.keys()):
						ans[i] = new_digit
						new_eqs.append(list(ans))
				hasfound, solution = cm.check_correct(equation,new_eqs,'+')
				if (hasfound == True):
					final_solution = solution
					return True, final_solution, '+' , False
	#change + to -, so a match should be added to digits (plus1) 	
	elif (oper == '+'):
		for i in range(6):
			new_eqs = []
			ans = digits[:]
			candidates = cm.plus.get(digits[i])
			for new_digit in candidates:
				if (new_digit in cm.str_to_num.keys()):
					ans[i] = new_digit
					new_eqs.append(list(ans))
			hasfound, solution = cm.check_correct(equation,new_eqs,'-')
			if (hasfound == True):
				final_solution = solution
				return True, final_solution, '-' , False
	
	return False, [], '', False

# check if correct answer can be achieved by moving a match within digits
def move_digit_1(equation, new_oper = None, new_digits = None):
	oper,__,digits = cm.parse_eq(equation)
	# check for call from move_oper_2
	if (new_oper != None):
		oper = new_oper
	
	if (new_digits != None):
		digits = new_digits[:]
	
	for i in range(6): 
		new_eqs = []
		ans = digits[:]	
		# change the pos of match within the same digit (same1)
		candidates_1 = cm.same.get(ans[i])
		for new_digit_1 in candidates_1:
			ans[i] = new_digit_1
			if set(ans).issubset(set(cm.str_to_num.keys())):
				new_eqs.append(list(ans))
		hasfound, solution = cm.check_correct(equation,new_eqs,oper)
		if (hasfound == True):
			final_solution = solution
			return True, final_solution, oper , False

		new_eqs = []
		ans = digits[:]
		# remove a match from one digit and added to a following digit (minus1-plus1)
		candidates_1 = cm.minus.get(ans[i])
		for new_digit_1 in candidates_1:
			ans[i] = new_digit_1
			for j in range(i+1,6):
				candidates_2 = cm.plus.get(ans[j])
				for new_digit_2 in candidates_2:
					ans[j] = new_digit_2
					if set(ans).issubset(set(cm.str_to_num.keys())):	
						new_eqs.append(list(ans))
				ans[j] = digits[j]

		hasfound, solution = cm.check_correct(equation,new_eqs,oper)	
		if (hasfound == True):
			final_solution = solution
			return True, final_solution, oper , False 

		new_eqs = []
		ans = digits[:]
		# add a match removed from a following digit to the previous digit (plus1-minus1)
		candidates_1 = cm.plus.get(ans[i])
		for new_digit_1 in candidates_1:
			ans[i] = new_digit_1
			for j in range(i+1,6):
				candidates_2 = cm.minus.get(ans[j])
				for new_digit_2 in candidates_2:
					ans[j] = new_digit_2
					if set(ans).issubset(set(cm.str_to_num.keys())):
						new_eqs.append(list(ans))
				ans[j] = digits[j]
		
		hasfound, solution = cm.check_correct(equation,new_eqs,oper)
		if (hasfound == True):
			final_solution = solution
			return True, final_solution, oper , False

	return False, [], '' , False