def read_data (database_file):
	database = open (database_file)
	X = []
	y = []
	line = database.readline ()

	while line:
		l = line.strip().split(',')
		x = l[1:]
		x = map (float, x)
		X.append (x)
		y.append (l[0].strip())
		line = database.readline()
	
	database.close ()
	return X,y