class Boosting( object ):
    def run( self ):
        raise NotImplementedError( "Should have implemented this" )
    
    def get_hypotheses( self ):
        raise NotImplementedError( "Should have implemented this" )
    
class WeakLearner( object ):
    def run( self ):
        raise NotImplementedError( "Should have implemented this" )