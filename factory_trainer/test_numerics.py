import unittest
import numpy as np
from .numerics import curves,nll,c_index,risk_rmst

class NumericalTests(unittest.TestCase):
    def test_exponential(self):
        # Exponential(rate=2): S(.5)=exp(-1), event NLL=1-log(2).
        a=curves([1.],[2.],[1.],[.5,1.])
        np.testing.assert_allclose(a['mixture_survival'],np.exp(-2*np.array([.5,1.])))
        np.testing.assert_allclose(nll([[1.],[1.]],[[2.],[2.]],[[1.],[1.]],[.5,.5],[0,1]),[1-np.log(2),1.])
    def test_mixture(self):
        a=curves([1.,2.],[.2,.1],[.3,.7],np.linspace(.01,20,500))
        np.testing.assert_allclose(a['mixture_pdf'],a['weighted_pdf'].sum(0))
        self.assertTrue((np.diff(a['mixture_survival'])<=1e-12).all())
    def test_rmst_and_concordance(self):
        risk=risk_rmst([[1.],[1.]],[[2.],[.5]],[[1.],[1.]],10)
        self.assertGreater(risk[0],risk[1])
        self.assertEqual(c_index([1,2],[0,0],[2,1])['c_index'],1.)
        self.assertEqual(c_index([1,2],[0,0],[1,2])['c_index'],0.)
        self.assertIsNone(c_index([1,2],[1,1],[2,1])['c_index'])

if __name__=='__main__':unittest.main()
