'''
	github.com/random_forests tutorial!!!
'''
import pandas as pd
from sklearn.model_selection import train_test_split
# Слздание маленького дата-сета
training_data = pd.read_csv('marks2.txt')
training_data = training_data.values
training_data,test_data = train_test_split(training_data,test_size=0.20)
# Заголовки
header = ['first','second','class']

def unique_vals(rows,col):
	'''Возвращает количество уникальных элементов колонки'''
	return set([row[col] for row in rows])

def class_counts(rows):
	'''
	Считает количество экземпляров обучающей выборки классов
	Возвращает словарь, где ключ - класс, а значение - количество
	'''
	counts = {}
	for row in rows:
		label = row[-1]
		if label not in counts:
			counts[label] = 0 
		counts[label] += 1
	return counts

def is_numeric(value):
	''' Возвращает True, если входные данные - число, иначе False '''
	return isinstance(value,int) or isinstance(value,float)

class Question():
	''' Вопрос используется для разделение данных.

	Класс записывает номер колонки и связанное с ним значение.
	Метод "match" задает вопрос и возвращает True,если ответ "да".
	'''
	def __init__(self,column,value):
		self.column = column
		self.value = value

	def match(self,example):
		# Сравнивает значение признака в example со значением признака в
		# вопросе(фактически, задает вопрос)
		val = example[self.column]
		if is_numeric(val):
			return val >= self.value
		else:
			return val == self.value

	def __repr__(self):
		# Вспомогательный метод, выводящий вопрос в читаемом формате
		condition = '=='
		if is_numeric(self.value):
			condition = '>='
		return 'Is %s %s %s?' % (
			header[self.column],condition,str(self.value))

def partition(rows,question):
	''' Разделяет данные.

	Для каждой строки в данных
	'''
	true_rows,false_rows = [], []
	for row in rows:
		if question.match(row):
			true_rows.append(row)
		else:
			false_rows.append(row)
	return true_rows,false_rows

def gini(rows):
	''' Считает индекс Джинни. '''
	counts = class_counts(rows)
	impurity = 1
	for lbl in counts:
		prob_of_lbl = counts[lbl] / float(len(rows))
		impurity -= prob_of_lbl ** 2
	return impurity

def information_gain(left,right,current_uncertainty):
	''' Считает увеличение информации.

	Неопределенность начального узла минус взвешенные неопределенности 
	двух дочерних узлов
	'''
	p = float(len(left)) / (len(left) + len(right))
	return current_uncertainty - p * gini(left) - (1-p) * gini(right)

def find_best_split(rows):
	''' Находит наилучший вопрос с помощью перебора каждого атрибута и его
		значения, рассчитывая при этом увеличение информации.
	'''
	best_gain = 0 # хранит лучшее значение inf_gain
	best_question = None # хранит лучши вопрос
	current_uncertainty = gini(rows) # неопределенность начального узла
	n_features = len(rows[0]) - 1 # количество колонок(признаков) - 1
	for col in range(n_features): # для каждого признака
		values = set([row[col] for row in rows]) # хранит уникальные значения
		for val in values: # для каждого значения признака
			question = Question(col,val) 
			# разделяет данные, основываясь на текущем вопросе
			true_rows,false_rows = partition(rows,question)
			# если данные не разделяются этим вопросом,
			# то пропускае это значение признака
			if len(true_rows) == 0 or len(false_rows) == 0:
				continue
			# вычисляем увеличение информации после разделения по вопросу
			gain = information_gain(true_rows,false_rows,current_uncertainty)
			# обновляется лучший gain и лучший question
			if gain >= best_gain:
				best_gain,best_question = gain,question
	return best_gain,best_question

class Leaf():
	''' Листовой узел классифицирует данные. Leaf - лист

	Хранит словарь с ключами-классами и значениями, показывающими
	сколько раз этот класс встречался в данных, дошедших до листового узла
	'''
	def __init__(self,rows):
		self.predictions = class_counts(rows)

class Decision_Node():
	''' Decision Node задает вопрос.

	Содержит ссылку на вопрос и на два дочерних узла
	'''
	def __init__(self,question,true_branch,false_branch):
		self.question = question
		self.true_branch = true_branch
		self.false_branch = false_branch

def build_tree(rows):
	''' Строит дерево.

	'''
	# находим лучший вопрос и лучшее увеличение
	gain,question = find_best_split(rows)
	# если увеличение 0, то мы не можем больше задавать вопросы, поэтому
	# возвращаем лист. (Базовый случай реукрсивной функции)
	if gain == 0:
		return Leaf(rows)
	# если мы дошли до сюда, то мы нашли полезный атрибут/значение
	# с помощью которого мы будем разделять данные
	true_rows,false_rows = partition(rows,question)
	# рекурсивно создаем true branch
	true_branch = build_tree(true_rows)
	# рекурсивно создаем false branch
	false_branch = build_tree(false_rows)
	# Возвращаем узел вопроса(Question Node)
	# Записывает лучший атрибут/значение и каким ветвям следовать(true/false)
	return Decision_Node(question,true_branch,false_branch)

def print_tree(node,spacing=''):
	''' Лучшая функция вывода дерева. '''

	# Базовый случай: мы достигли листа
	if isinstance(node,Leaf):
		print(spacing + 'Predict',node.predictions)
		return
	# Выводим вопрос этого узла
	print(spacing + str(node.question))
	# Вызываем эту функцию рекурсивно на true branch
	print(spacing + '--> True:')
	print_tree(node.true_branch,spacing + ' ')
	# Вызываем эту функцию рекурсивно на false branch
	print(spacing + '--> False:')
	print_tree(node.false_branch,spacing + ' ')

my_tree = build_tree(training_data)
print_tree(my_tree)

def classify(row,node):

	# Базовый случай, мы достигли листа
	if isinstance(node,Leaf):
		return node.predictions
	if node.question.match(row):
		return classify(row,node.true_branch)
	else:
		return classify(row,node.false_branch)

def print_leaf(counts):
	''' Другой вид вывода листа '''
	total = sum(counts.values())
	probs = {}
	for lbl in counts.keys():
		probs[lbl] = str(int(counts[lbl]/total * 100)) + '%'
	return probs

for row in test_data:
	print("Actual: %s. Predicted: %s" %
		(row[-1],print_leaf(classify(row,my_tree))))
	