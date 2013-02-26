import json, inspect, os, sqlite3

class BaseFeatureFactory(object):
	def __init__(self,data,db_path=None):
		self.data = data[:,:]
		self.cls_path = str(os.path.realpath(inspect.getfile(self.__class__)))
		self.cls_name = str(self.__class__.__name__)
		if db_path:
			self.db_path = db_path
			self.conn = sqlite3.connect(self.db_path)
			self.cursor = self.conn.cursor()
		
	def blueprint(self, *args):
		raise NotImplementedError("Error : "+
						"Should have implemented a blueprint method")
	
	def make(self):
		raise NotImplementedError("Error : " +
						"Should have implemented a make method")
	
	def insert(self, *args):
		argstr = json.dumps(args)
		statement = "INSERT INTO features VALUES ('" +self.cls_path+"','"+self.cls_name+"','"+argstr+"')"
		self.cursor.execute(statement)
	
	def finalize(self):
		rc = self.cursor.lastrowid
		self.conn.commit()
		self.conn.close()
		return rc
	
class DefaultFeatureFactory(BaseFeatureFactory):
	def blueprint(self,attr):
		return self.data[:,attr]
	
	def make(self):
		for k in range(0, self.data.shape[1]-2):
			attribute = k
			self.insert(attribute)