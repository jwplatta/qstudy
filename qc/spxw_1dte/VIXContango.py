from AlgorithmImports import *

class VIXContango:
    """Manages VIX term structure analysis for contango/backwardation"""
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.vix = algorithm.vix
        self.vix1d = algorithm.vix1d
        self.vix9d = algorithm.vix9d
    
    def is_in_contango(self):
        """
        Check if VIX term structure is in contango
        Contango: VIX1D < VIX9D < VIX (30-day)
        """
        try:
            vix_price = self.algorithm.securities[self.vix].price
            vix1d_price = self.algorithm.securities[self.vix1d].price
            vix9d_price = self.algorithm.securities[self.vix9d].price
            
            if not all([vix_price, vix1d_price, vix9d_price]):
                self.algorithm.debug(f"Invalid VIX prices: VIX1D={vix1d_price}, VIX9D={vix9d_price}, VIX={vix_price}")
                return False
            
            is_contango = (vix1d_price < vix9d_price) and (vix9d_price < vix_price)

            if is_contango:
                self.algorithm.debug(f"VIX CONTANGO: VIX1D={vix1d_price:.2f} < VIX9D={vix9d_price:.2f} < VIX={vix_price:.2f}")
                return True
            else:
                self.algorithm.debug(f"VIX BACKWARDATION: VIX1D={vix1d_price:.2f}, VIX9D={vix9d_price:.2f}, VIX={vix_price:.2f}")
                return False
        except Exception as e:
            self.algorithm.debug(f"Error checking VIX contango: {str(e)}")
            return False
    
    def get_vix_term_structure(self):
        """
        Get current VIX term structure values
        Returns dict with VIX prices and spreads
        """
        try:
            vix_price = self.algorithm.securities[self.vix].price
            vix1d_price = self.algorithm.securities[self.vix1d].price
            vix9d_price = self.algorithm.securities[self.vix9d].price
            
            return {
                "vix_1d": vix1d_price,
                "vix_9d": vix9d_price,
                "vix_30d": vix_price,
                "spread_1d_9d": vix9d_price - vix1d_price,
                "spread_9d_30d": vix_price - vix9d_price,
                "spread_1d_30d": vix_price - vix1d_price,
            }
        except:
            return None
