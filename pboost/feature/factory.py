import ujson, inspect, os, sqlite3

class BaseFeatureFactoryManager(object):
    def __init__(self,data_handler,db_path):
        self.data_handler = data_handler
        self.cls_path = str(os.path.realpath(inspect.getfile(self.__class__)))
        self.cls_name = str(self.__class__.__name__)
        self.db_path = db_path
        self.entries = list()
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
            
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

class BaseFeatureFactory(object):
    def __init__(self):
        self.data = None
        self.cls_path = str(os.path.realpath(inspect.getfile(self.__class__)))
        self.cls_name = str(self.__class__.__name__)
    
    def load_data(self,data_handler,*args):
        ''' This method can be customized wrt memory req '''
        self.data = data_handler[...]

    def blueprint(self, *args):
        raise NotImplementedError("Error : "+
                        "Should have implemented a blueprint method")
           
class DefaultFeatureFactoryManager(BaseFeatureFactoryManager):
    
    def produce(self):
        for k in range(0, self.data.shape[1]-2):
    	   attribute = k
    	   self.make(attribute)
           
class DefaultFeatureFactory(BaseFeatureFactory):
           
    def blueprint(self,attr):
        return self.data[:,attr]
