def testParams():
	file = open('tt06_GPU_SA_Parameters.txt','r')
	count = 0
	for line in file:
		++count
		if math.isNan(line):
			print 'Nan found at line number', count
			return False
	print count
	return True
def testVolgate():
	file = open('tt06_GPU_Voltage.txt','r')
	for line in file:
		++count
		if math.isNan(line):
			print 'Nan found at line number', count
			return False
		return True	

print testParams()
