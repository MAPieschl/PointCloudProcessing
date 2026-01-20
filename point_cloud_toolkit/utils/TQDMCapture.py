class TQDMCapture:
    def __init__( self ):
        self.last = ""

    def write( self, msg ) -> None:
        self.last = msg.strip()

    def flush( self ) -> None:
        pass

    def get_percent( self ) -> int | None:
        if( "%" in self.last ):
            end_idx = self.last.find( "%" )
            try:
                return int( self.last[:end_idx] )
            except:
                return None
        
        else:
            return None