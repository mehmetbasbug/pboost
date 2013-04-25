import ujson, inspect, os, sqlite3

class BaseFeatureFactory(object):
    def __init__(self,data,db_path=None):
        self.data = data[:,:]
        self.cls_path = str(os.path.realpath(inspect.getfile(self.__class__)))
        self.cls_name = str(self.__class__.__name__)
        if db_path:
            self.db_path = db_path
            self.entries = list()
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        
    def blueprint(self, *args):
        raise NotImplementedError("Error : "+
                        "Should have implemented a blueprint method")
    
    def produce(self):
        raise NotImplementedError("Error : " +
                        "Should have implemented a produce method")
    
    def make(self, *args):
        """
        DO NOT OVERRIDE THIS METHOD
        """
        argstr = ujson.dumps(args)
        self.entries.append([self.cls_path,self.cls_name,argstr])
    
    def finalize(self):
        self.cursor.executemany('INSERT INTO features VALUES (?,?,?)', self.entries)
        rc = self.cursor.rowcount
        self.conn.commit()
        self.conn.close()
        return rc
       
class DefaultFeatureFactory(BaseFeatureFactory):
	def blueprint(self,attr):
		return self.data[:,attr]
	
	def produce(self):
		for k in range(0, self.data.shape[1]-2):
			attribute = k
			self.make(attribute)