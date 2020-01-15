# -*- coding: utf-8 -*-
# This file contains gui of the whole project
"""在命令行里运行即可生成GUI程序"""
import common as cm
import matches1
import matches2

import sys
import sip
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QLabel, QGridLayout, QPushButton, QComboBox, QLineEdit, QMessageBox
from PyQt5.QtGui import QPixmap

# mapping images
class digit_num():
	def __init__(self):
		super().__init__()
	
	def path_dict(self):
		to_path = { 
			'0000000':"resources/empty.png", 
			'1110111':"resources/0.png", 
			'0010010':"resources/1_1.png", 
			'0100100':"resources/1_2.png", 
			'1011101':"resources/2.png", 
			'1011011':"resources/3.png", 
			'0111010':"resources/4.png", 
			'1101011':"resources/5.png", 
			'1101111':"resources/6_1.png", 
			'0101111':"resources/6_2.png", 
			'1010010':"resources/7.png", 
			'1111111':"resources/8.png", 
			'1111011':"resources/9_1.png", 
			'1111010':"resources/9_2.png", 
			'+':"resources/+.png", 
			'-':"resources/-.png", 
			'*':"resources/*.png", 
			'=':"resources/=.png" }	
		return to_path

	def to_pics(self,path):
		d_raw = QPixmap(path)
		d = QLabel()
		d.setPixmap(d_raw.scaledToHeight(100))
		return d

# This class generates the main window of the app
class MainUI(QWidget):	
	def __init__(self):
		super(MainUI, self).__init__()
		self.initUI()
	
	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def initUI(self):
		self.resize(800, 600)
		self.center()
		self.setWindowTitle('Matches Equation')

		grid = QGridLayout()
		self.setLayout(grid)
		for i in range(5):
			if i == 0:
				grid.setRowMinimumHeight(i, 30)
			else: 
				grid.setRowMinimumHeight(i, 100)
		for j in range(10):
			if j == 9:
				grid.setColumnMinimumWidth(j, 100)
			else:
				grid.setColumnMinimumWidth(j, 60)
		
		ori_eq_label = QLabel('原算式')
		grid.addWidget(ori_eq_label, 1, 0, 2, 1)
		sol_eq_label = QLabel('变换后算式')
		grid.addWidget(sol_eq_label, 3, 0, 2, 1)

		one_button = QPushButton('移动一根', self)
		one_button.clicked.connect(self.to_secUI1)
		one_button.resize(one_button.sizeHint())
		grid.addWidget(one_button, 1, 9)

		two_button = QPushButton('移动两根', self)
		two_button.clicked.connect(self.to_secUI1)
		two_button.resize(two_button.sizeHint())
		grid.addWidget(two_button, 2, 9)

		equal_button = QPushButton('等式转变', self)
		equal_button.clicked.connect(self.to_secUI2)
		equal_button.resize(equal_button.sizeHint())
		grid.addWidget(equal_button, 3, 9)

	def to_secUI1(self):
		self.hide()
		self.win2 = secUI1()
		self.win2.show()

	def to_secUI2(self):
		self.hide()
		self.win3 = secUI2()
		self.win3.show()

# This class generates the window of the app when you choose to move one/two matches
class secUI1(QWidget):
	def __init__(self):
		super(secUI1,self).__init__()
		self.initUI()
	
	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())
	
	def initUI(self):
		self.resize(800, 600)
		self.center()
		self.setWindowTitle('Matches Equation')	
		self.send_from = self.sender()

		self.grid = QGridLayout()
		self.setLayout(self.grid)
		for i in range(5):
			if i == 0:
				self.grid.setRowMinimumHeight(i, 30)
			else: 
				self.grid.setRowMinimumHeight(i, 100)	
		for j in range(10):
			if j == 9:
				self.grid.setColumnMinimumWidth(j, 100)
			else:
				self.grid.setColumnMinimumWidth(j, 60)

		ori_eq_label = QLabel('原算式')
		self.grid.addWidget(ori_eq_label, 1, 0, 2, 1)
		sol_eq_label = QLabel('变换后算式')
		self.grid.addWidget(sol_eq_label, 3, 0, 2, 1)

		self.load_button = QPushButton('从题库加载', self)
		self.load_button.clicked.connect(self.load_eq_from_file)
		self.load_button.resize(self.load_button.sizeHint())
		self.grid.addWidget(self.load_button, 0, 9)

		self.input_button = QPushButton('输入算式', self)
		self.input_button.clicked.connect(self.input_eq_from_line)
		self.input_button.resize(self.input_button.sizeHint())
		self.grid.addWidget(self.input_button, 1, 9)
		
		self.random_button = QPushButton('随机生成', self)
		self.random_button.clicked.connect(self.random_generate_eq)
		self.random_button.resize(self.random_button.sizeHint())
		self.grid.addWidget(self.random_button, 2, 9)
		
		self.back_button = QPushButton('返回', self)
		self.back_button.clicked.connect(self.to_MainUI)
		self.back_button.resize(self.back_button.sizeHint())
		self.grid.addWidget(self.back_button, 4, 9)
	
	# when choosing to load problem sets from existing file
	def load_eq_from_file(self):
		self.send_from_2 = self.sender()
		self.load_button.hide()
		self.input_button.hide()
		self.random_button.hide()

		prompt = QLabel('请选择一个题库中的算式')
		self.grid.addWidget(prompt, 0, 0, 1, 2)
		
		if (self.send_from.text() == '移动一根'):
			with open('resources/equations.txt') as eq:
				equations = [line[:-1] for line in eq]		
			self.combo = QComboBox(self)
			for equation in equations:
				 self.combo.addItem(equation)
			self.grid.addWidget(self.combo, 0, 2, 1, 2)	
			self.combo.activated[str].connect(self.compute_1)	
		
		elif (self.send_from.text() == '移动两根'):
			with open('resources/equations2.txt') as eq:
				equations = [line[:-1] for line in eq]	
			self.combo = QComboBox(self)
			for equation in equations:
				 self.combo.addItem(equation)
			self.grid.addWidget(self.combo, 0, 2, 1, 2)	
			self.combo.activated[str].connect(self.compute_2)

	# when choosing to get a problem by user input
	def input_eq_from_line(self):
		self.send_from_2 = self.sender()
		self.load_button.hide()
		self.input_button.hide()
		self.random_button.hide()	

		prompt = QLabel('请输入一个算式')
		self.grid.addWidget(prompt, 0, 0, 1, 2)

		self.line = QLineEdit(self)
		self.grid.addWidget(self.line, 0, 2, 1, 2)
		
		if (self.send_from.text() == '移动一根'):
			self.line.returnPressed.connect(self.compute_1)
		elif (self.send_from.text() == '移动两根'):
			self.line.returnPressed.connect(self.compute_2)		

	# when choosing to generate a random problem automatically
	def random_generate_eq(self):
		self.send_from_2 = self.sender()
		self.load_button.hide()
		self.input_button.hide()
		self.random_button.hide()

		prompt = QLabel('按右键随机生成算式')
		self.grid.addWidget(prompt, 0, 0, 1, 2)

		gen_button = QPushButton('生成算式', self)
		gen_button.resize(gen_button.sizeHint())
		gen_button.resize(gen_button.sizeHint())
		self.grid.addWidget(gen_button, 0, 2)

		if (self.send_from.text() == '移动一根'):
			gen_button.clicked.connect(self.compute_1)
		elif (self.send_from.text() == '移动两根'):
			gen_button.clicked.connect(self.compute_2)
	
	# when choosing to move one match
	def compute_1(self):
		if (self.send_from_2.text() == '从题库加载'):
			equation_selected = self.combo.currentText()
		elif (self.send_from_2.text() == '输入算式'):
			equation_selected = self.line.text()
		elif(self.send_from_2.text() == '随机生成'):
			equation_selected = cm.question_generator()
		
		ori_oper,__,ori_digits = cm.parse_eq(equation_selected)
		self.load_pics(ori_oper,ori_digits,1)

		found, solution, oper, isreversed = matches1.move_oper_1(equation_selected)
		if (found == True):	
			self.load_pics(oper,solution,3,isreversed)
		else:
			found, solution, oper, isreversed = matches1.move_digit_1(equation_selected)
			if (found == True):	
				self.load_pics(oper,solution,3,isreversed)
			else:
				QMessageBox.information(self, "无解信息", "该算式不能通过移动一根火柴变成等式!",
                                QMessageBox.Ok)
		
		self.clear_button = QPushButton('清空', self)
		self.clear_button.clicked.connect(self.clear_result)
		self.clear_button.resize(self.clear_button.sizeHint())
		self.grid.addWidget(self.clear_button, 0, 4)
	
	# when choosing to move two matches
	def compute_2(self):
		if (self.send_from_2.text() == '从题库加载'):
			equation_selected = self.combo.currentText()
		elif (self.send_from_2.text() == '输入算式'):
			equation_selected = self.line.text()
		elif(self.send_from_2.text() == '随机生成'):
			equation_selected = cm.question_generator()
		
		ori_oper,__,ori_digits = cm.parse_eq(equation_selected)
		self.load_pics(ori_oper,ori_digits,1)

		found, solution, oper, isreversed = matches2.move_oper_2(equation_selected)
		if (found == True):	
			self.load_pics(oper,solution,3,isreversed)
		else:
			found, solution, oper, isreversed = matches2.move_digit_2(equation_selected)
			if (found == True):	
				self.load_pics(oper,solution,3,isreversed)
			else:
				QMessageBox.information(self, "无解信息", "该算式不能通过移动两根火柴变成等式!",
                                QMessageBox.Ok)
		
		self.clear_button = QPushButton('清空', self)
		self.clear_button.clicked.connect(self.clear_result)
		self.clear_button.resize(self.clear_button.sizeHint())
		self.grid.addWidget(self.clear_button, 0, 4)

	def load_pics(self,oper,digits,start_row,reverse=False):
		d = digit_num()
		to_path = d.path_dict()	
		for i in range(1,9):
			if (i < 3):
				path = to_path[digits[i-1]] 
			elif (i == 3):
				path = to_path[oper] if (reverse == False) else to_path['=']
			elif (i > 3) and (i < 6):
				path = to_path[digits[i-2]]
			elif (i == 6):
				path = to_path['='] if (reverse == False) else to_path[oper]
			else:
				path = to_path[digits[i-3]]
			pic = d.to_pics(path)
			self.grid.addWidget(pic, start_row, i, 2, 1)
	
	def clear_result(self):
		for i in range(1,5):
			for j in range(1,9):
				pic = self.grid.itemAtPosition(i,j)
				if pic is not None: 
					self.grid.removeWidget(pic.widget())
					sip.delete(pic.widget())	
		self.grid.removeWidget(self.clear_button)
		sip.delete(self.clear_button)

	def to_MainUI(self):
		self.hide()
		self.win1 = MainUI()
		self.win1.show()

# This class generates the window of the app when you choose to check the hardship changing one equation from another
class secUI2(QWidget):
	def __init__(self):
		super(secUI2,self).__init__()
		self.initUI()
	
	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())
	
	def initUI(self):
		self.resize(800, 600)
		self.center()
		self.setWindowTitle('Matches Equation')

		self.grid = QGridLayout()
		self.setLayout(self.grid)

		for i in range(5):
			if i == 0:
				self.grid.setRowMinimumHeight(i, 30)
			else: 
				self.grid.setRowMinimumHeight(i, 100)
		
		for j in range(10):
			if j == 9:
				self.grid.setColumnMinimumWidth(j, 100)
			else:
				self.grid.setColumnMinimumWidth(j, 60)

		ori_eq_label = QLabel('原算式')
		self.grid.addWidget(ori_eq_label, 1, 0, 2, 1)
		sol_eq_label = QLabel('变换后算式')
		self.grid.addWidget(sol_eq_label, 3, 0, 2, 1)
		
		self.pool_button = QPushButton('等式库加载', self)
		self.pool_button.clicked.connect(self.load_eq_pool)
		self.pool_button.resize(self.pool_button.sizeHint())
		self.grid.addWidget(self.pool_button, 3, 9)
		
		self.back_button = QPushButton('返回', self)
		self.back_button.clicked.connect(self.to_MainUI)
		self.back_button.resize(self.back_button.sizeHint())
		self.grid.addWidget(self.back_button, 4, 9)

	# load existing equations (randomly generated and eliminate duplication)
	def load_eq_pool(self):
		self.pool_button.hide()
		
		with open('resources/equations637.txt') as eq:
			self.equations = [line[:-1] for line in eq]
				
		prompt = QLabel('请选择一个题库中的算式')
		self.grid.addWidget(prompt, 0, 0, 1, 2)

		self.combo = QComboBox(self)
		for equation in self.equations:
			self.combo.addItem(equation)
		self.grid.addWidget(self.combo, 0, 2, 1, 2)	
		self.combo.activated[str].connect(self.find_solution)
	
	# check for possible solutions
	def find_solution(self):
		equation_selected = self.combo.currentText()	
		input_oper,__,input_digits = cm.parse_eq(equation_selected)
		self.load_pics(input_oper,input_digits,1)

		input_match_cnt, input_eq_str, input_oper_det = cm.matching(input_oper, input_digits)
		self.solution_stack = []

		for equation in self.equations:
			oper,__,digits = cm.parse_eq(equation)
			match_cnt, eq_str, oper_det = cm.matching(oper,digits)
			if (match_cnt == input_match_cnt):
				hardship = cm.hardship(input_eq_str, eq_str, input_oper_det, oper_det)
				self.solution_stack.append((oper,digits,hardship))

		self.prompt_1 = QLabel('难度系数')
		self.grid.addWidget(self.prompt_1, 0, 6)
		self.display = QLabel()
		self.grid.addWidget(self.display, 0, 7)

		self.prompt_2 = QLabel('剩余答案数')
		self.grid.addWidget(self.prompt_2, 0, 8)
		self.countdown = QLabel()
		self.grid.addWidget(self.countdown, 0, 9)

		self.display_button = QPushButton('显示结果', self)
		self.display_button.clicked.connect(self.display_result)
		self.display_button.resize(self.display_button.sizeHint())
		self.grid.addWidget(self.display_button, 0, 4)

		self.next_button = QPushButton('清除', self)
		self.next_button.clicked.connect(self.next_result)
		self.next_button.resize(self.next_button.sizeHint())
		self.grid.addWidget(self.next_button, 0, 5)

	def load_pics(self,oper,digits,start_row,reverse=False):
		d = digit_num()
		to_path = d.path_dict()	
		for i in range(1,9):
			if (i < 3):
				path = to_path[digits[i-1]] 
			elif (i == 3):
				path = to_path[oper] if (reverse == False) else to_path['=']
			elif (i > 3) and (i < 6):
				path = to_path[digits[i-2]]
			elif (i == 6):
				path = to_path['='] if (reverse == False) else to_path[oper]
			else:
				path = to_path[digits[i-3]]
			pic = d.to_pics(path)
			self.grid.addWidget(pic, start_row, i, 2, 1)
	
	# display all possible solutions and hardships
	def display_result(self):
		if (self.solution_stack != []):
			self.solution = self.solution_stack.pop()
			(oper, digits, hardship) = self.solution
			self.load_pics(oper,digits,3)
			self.display.setNum(hardship)
			self.countdown.setNum(len(self.solution_stack))
		else:
			QMessageBox.information(self, "显示完毕", "所有结果显示完毕!",
                                QMessageBox.Ok)

	def next_result(self):
		for i in range(3,5):
			for j in range(1,9):
				pic = self.grid.itemAtPosition(i,j)
				if pic is not None: 
					self.grid.removeWidget(pic.widget())
					sip.delete(pic.widget())
		self.display.setText('')
		if (self.solution_stack == []):
			for i in range(1,5):
				for j in range(1,9):
					pic = self.grid.itemAtPosition(i,j)
					if pic is not None: 
						self.grid.removeWidget(pic.widget())
						sip.delete(pic.widget())

	
	def to_MainUI(self):
		self.hide()
		self.win1 = MainUI()
		self.win1.show()

if __name__ == '__main__':
	app = QApplication(sys.argv)
	win1 = MainUI()
	win1.show()
	sys.exit(app.exec_())